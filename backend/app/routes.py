import jwt
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import PyPDF2
import docx
import re
import difflib

from .models import SessionLocal, User, DetectionResult, PlagiarismLog
from .ml.model_helper import detect_plagiarism, generate_highlighted_diff, ensemble_similarity

api_bp = Blueprint("api", __name__)
JWT_SECRET = "vericheck-secret-key-2024-secure"

# FREE TIER LIMIT
FREE_WORD_LIMIT = 200

# Helper text extractors
def extract_text_from_pdf(stream):
    reader = PyPDF2.PdfReader(stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(stream):
    doc = docx.Document(stream)
    text = [p.text for p in doc.paragraphs]
    return "\n".join(text)

def count_words(text):
    return len(text.split())

def count_chars(text):
    return len(text)

def get_user_from_token():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        db = SessionLocal()
        user = db.query(User).filter(User.id == data["user_id"]).first()
        return user
    except:
        return None

# --- SINGLE TEXT PLAGIARISM CHECK ---
@api_bp.route("/free-check-single", methods=["POST"])
def free_check_single():
    """Free single text check (up to 200 words)"""
    text = ""

    # Handle file upload (multipart/form-data)
    if request.content_type and "multipart/form-data" in request.content_type:
        file = request.files.get("file")
        if file:
            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file.stream)
            elif file.filename.endswith(".docx"):
                text = extract_text_from_docx(file.stream)
            else:
                text = file.read().decode("utf-8", errors="ignore")
        else:
            text = request.form.get("text", "")
    else:
        data = request.json or {}
        text = data.get("text", "")

    word_count = count_words(text)

    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    if word_count > FREE_WORD_LIMIT:
        return jsonify({
            "error": f"Free tier allows up to {FREE_WORD_LIMIT} words only.",
            "word_count": word_count,
            "free_limit": FREE_WORD_LIMIT,
            "requires_signup": True
        }), 403

    # Run ensemble plagiarism detection
    result = detect_plagiarism(text)

    return jsonify({
        "similarity": result["similarity"],
        "level": result["level"],
        "highlighted_html": result["highlighted_html"],
        "sentence_analysis": result["sentence_analysis"][:10],  # send first 10 sentences
        "metrics": result["metrics"],
        "word_count": result["word_count"],
        "char_count": result["char_count"],
        "explanation": result["explanation"]
    }), 200


@api_bp.route("/detect-single", methods=["POST"])
def detect_single():
    """Single text check for authenticated users"""
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    text = ""
    input_type = "text"
    source_info = "Single Text Check"

    # Handle file upload (multipart/form-data)
    if request.content_type and "multipart/form-data" in request.content_type:
        file = request.files.get("file")
        if file:
            filename = file.filename
            source_info = filename
            input_type = "file"
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(file.stream)
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(file.stream)
            else:
                text = file.read().decode("utf-8", errors="ignore")
        else:
            text = request.form.get("text", "")
    else:
        # Handle JSON payload
        data = request.json or {}
        text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    word_count = count_words(text)

    # Run ensemble plagiarism detection
    result = detect_plagiarism(text)

    # Save detection
    db = SessionLocal()
    record = DetectionResult(
        user_id=user.id,
        input_text=text[:500],
        similarity=result["similarity"],
        highlighted_html=result["highlighted_html"],
        source_info=source_info,
        input_type=input_type,
        word_count_a=word_count,
        chars_a=count_chars(text)
    )
    db.add(record)
    db.commit()

    # Log
    log = PlagiarismLog(
        user_id=user.id,
        action="CHECK",
        input_type="text",
        input_chars=count_chars(text),
        word_count=word_count,
        result=f"Score: {result['similarity']:.2%} | Level: {result['level']}",
        similarity_score=result["similarity"]
    )
    db.add(log)
    db.commit()

    return jsonify({
        "id": record.id,
        "similarity": result["similarity"],
        "level": result["level"],
        "highlighted_html": result["highlighted_html"],
        "sentence_analysis": result["sentence_analysis"][:10],
        "metrics": result["metrics"],
        "word_count": result["word_count"],
        "char_count": result["char_count"],
        "explanation": result["explanation"],
        "created_at": record.created_at.isoformat()
    }), 200


# --- FREE TIER CHECK ---
@api_bp.route("/free-check", methods=["POST"])
def free_check():
    """
    Free plagiarism check without login (up to 200 words).
    """
    data = request.json or {}
    text_a = data.get("text_a", "")
    text_b = data.get("text_b", "")

    word_count_a = count_words(text_a)
    word_count_b = count_words(text_b)

    if not text_a.strip() or not text_b.strip():
        return jsonify({"error": "Both texts are required"}), 400

    if word_count_a > FREE_WORD_LIMIT or word_count_b > FREE_WORD_LIMIT:
        return jsonify({
            "error": f"Free tier allows up to {FREE_WORD_LIMIT} words only. Please sign up for unlimited checks.",
            "word_count_a": word_count_a,
            "word_count_b": word_count_b,
            "free_limit": FREE_WORD_LIMIT,
            "requires_signup": True
        }), 403

    score = ensemble_similarity(text_a, text_b)
    highlighted = generate_highlighted_diff(text_a, text_b)

    return jsonify({
        "similarity": score,
        "highlighted_html": highlighted,
        "word_count_a": word_count_a,
        "word_count_b": word_count_b,
        "chars_a": count_chars(text_a),
        "chars_b": count_chars(text_b),
        "is_free_tier": True,
        "source_info": "Free Quick Check"
    }), 200

# --- AUTHENTICATION ---
@api_bp.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "Missing required fields: username, email, password"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = SessionLocal()
    if db.query(User).filter(User.username == username).first():
        return jsonify({"error": "Username already taken"}), 409
    if db.query(User).filter(User.email == email).first():
        return jsonify({"error": "Email already registered"}), 409

    pwd_hash = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=pwd_hash)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Log the registration
    log = PlagiarismLog(
        user_id=new_user.id,
        action="REGISTER",
        input_type="text",
        input_chars=0,
        word_count=0,
        result=f"New user registered: {username}",
        ip_address=request.remote_addr
    )
    db.add(log)
    db.commit()

    token = jwt.encode({
        "user_id": new_user.id,
        "exp": datetime.utcnow() + timedelta(days=30),
        "iat": datetime.utcnow()
    }, JWT_SECRET, algorithm="HS256")

    return jsonify({
        "token": token,
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "is_admin": new_user.is_admin
        }
    }), 201

@api_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = SessionLocal()
    user = db.query(User).filter(User.email == email).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401

    # Log the login
    log = PlagiarismLog(
        user_id=user.id,
        action="LOGIN",
        input_type="text",
        input_chars=0,
        word_count=0,
        result="User logged in",
        ip_address=request.remote_addr
    )
    db.add(log)
    db.commit()

    token = jwt.encode({
        "user_id": user.id,
        "exp": datetime.utcnow() + timedelta(days=30),
        "iat": datetime.utcnow()
    }, JWT_SECRET, algorithm="HS256")

    return jsonify({
        "token": token,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin
        }
    }), 200

@api_bp.route("/auth/me", methods=["GET"])
def get_me():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin
    }), 200

# --- DETECTION ENDPOINT ---
@api_bp.route("/detect", methods=["POST"])
def detect():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized. Please login."}), 401

    text_a, text_b = "", ""
    source_name = "Quick Comparison"
    input_type = "text"

    # Handle file uploads
    if "file_a" in request.files or "file_b" in request.files:
        file_a = request.files.get("file_a")
        file_b = request.files.get("file_b")
        input_type = "file"

        if file_a:
            source_name = f"{file_a.filename}"
            if file_a.filename.endswith(".pdf"):
                text_a = extract_text_from_pdf(file_a.stream)
            elif file_a.filename.endswith(".docx"):
                text_a = extract_text_from_docx(file_a.stream)
            else:
                text_a = file_a.read().decode("utf-8", errors="ignore")

        if file_b:
            if source_name == "Quick Comparison":
                source_name = f"{file_b.filename}"
            else:
                source_name = f"{source_name} vs {file_b.filename}"
            if file_b.filename.endswith(".pdf"):
                text_b = extract_text_from_pdf(file_b.stream)
            elif file_b.filename.endswith(".docx"):
                text_b = extract_text_from_docx(file_b.stream)
            else:
                text_b = file_b.read().decode("utf-8", errors="ignore")
    else:
        json_data = request.json or {}
        text_a = json_data.get("text_a", "")
        text_b = json_data.get("text_b", "")
        source_name = json_data.get("source_info", "Quick Comparison")

    if not text_a.strip() or not text_b.strip():
        return jsonify({"error": "Both comparison texts must have content."}), 400

    # Use ensemble detection on text_a (text_b as reference)
    from .ml.model_helper import ensemble_similarity
    scores = ensemble_similarity(text_a, text_b)
    score = scores.get("ensemble", scores.get("seqmatch", 0))
    highlighted = generate_highlighted_diff(text_a, text_b)

    wc_a = count_words(text_a)
    wc_b = count_words(text_b)

    # Save detection
    db = SessionLocal()
    record = DetectionResult(
        user_id=user.id,
        input_text=text_a[:500] + "...",
        similarity=score,
        highlighted_html=highlighted,
        source_info=source_name,
        input_type=input_type,
        word_count_a=wc_a,
        word_count_b=wc_b,
        chars_a=count_chars(text_a),
        chars_b=count_chars(text_b)
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Log the detection
    log = PlagiarismLog(
        user_id=user.id,
        action="DETECT",
        input_type=input_type,
        input_chars=count_chars(text_a) + count_chars(text_b),
        word_count=wc_a + wc_b,
        result=f"Similarity: {score:.2%}",
        similarity_score=score,
        ip_address=request.remote_addr
    )
    db.add(log)
    db.commit()

    return jsonify({
        "id": record.id,
        "similarity": score,
        "highlighted_html": highlighted,
        "source_info": source_name,
        "word_count_a": wc_a,
        "word_count_b": wc_b,
        "chars_a": count_chars(text_a),
        "chars_b": count_chars(text_b),
        "created_at": record.created_at.isoformat()
    }), 200

# --- USER DASHBOARD ---
@api_bp.route("/dashboard", methods=["GET"])
def get_dashboard():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    db = SessionLocal()

    # Get all detections
    scans = db.query(DetectionResult).filter(
        DetectionResult.user_id == user.id
    ).order_by(DetectionResult.created_at.desc()).limit(50).all()

    # Calculate stats
    total_scans = len(scans)
    if total_scans > 0:
        avg_similarity = sum(s.similarity for s in scans) / total_scans
        high_risk = sum(1 for s in scans if s.similarity > 0.7)
        medium_risk = sum(1 for s in scans if 0.4 < s.similarity <= 0.7)
        low_risk = sum(1 for s in scans if s.similarity <= 0.4)
        total_words = sum((s.word_count_a or 0) + (s.word_count_b or 0) for s in scans)
    else:
        avg_similarity = 0
        high_risk = medium_risk = low_risk = total_words = 0

    return jsonify({
        "total_scans": total_scans,
        "average_similarity": round(avg_similarity, 4),
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": low_risk,
        "total_words_checked": total_words,
        "recent_scans": [{
            "id": s.id,
            "similarity": s.similarity,
            "source_info": s.source_info,
            "input_type": s.input_type,
            "word_count": (s.word_count_a or 0) + (s.word_count_b or 0),
            "created_at": s.created_at.isoformat()
        } for s in scans[:10]]
    }), 200

# --- USER REPORT HISTORY ---
@api_bp.route("/scans", methods=["GET"])
def get_scans():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    skip = int(request.args.get("skip", 0))
    limit = int(request.args.get("limit", 20))

    db = SessionLocal()
    scans = db.query(DetectionResult).filter(
        DetectionResult.user_id == user.id
    ).order_by(DetectionResult.created_at.desc()).offset(skip).limit(limit).all()

    return jsonify([{
        "id": s.id,
        "similarity": s.similarity,
        "source_info": s.source_info,
        "input_type": s.input_type,
        "word_count_a": s.word_count_a,
        "word_count_b": s.word_count_b,
        "chars_a": s.chars_a,
        "chars_b": s.chars_b,
        "created_at": s.created_at.isoformat()
    } for s in scans]), 200

@api_bp.route("/scans/<int:scan_id>", methods=["GET"])
def get_scan_by_id(scan_id):
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    db = SessionLocal()
    scan = db.query(DetectionResult).filter(
        DetectionResult.id == scan_id,
        DetectionResult.user_id == user.id
    ).first()

    if not scan:
        return jsonify({"error": "Scan not found"}), 404

    return jsonify({
        "id": scan.id,
        "similarity": scan.similarity,
        "highlighted_html": scan.highlighted_html,
        "source_info": scan.source_info,
        "input_type": scan.input_type,
        "word_count_a": scan.word_count_a,
        "word_count_b": scan.word_count_b,
        "chars_a": scan.chars_a,
        "chars_b": scan.chars_b,
        "created_at": scan.created_at.isoformat()
    }), 200

@api_bp.route("/scans/<int:scan_id>", methods=["DELETE"])
def delete_scan(scan_id):
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    db = SessionLocal()
    scan = db.query(DetectionResult).filter(
        DetectionResult.id == scan_id,
        DetectionResult.user_id == user.id
    ).first()

    if not scan:
        return jsonify({"error": "Scan not found"}), 404

    db.delete(scan)

    # Log deletion
    log = PlagiarismLog(
        user_id=user.id,
        action="DELETE",
        input_type="text",
        input_chars=0,
        word_count=0,
        result=f"Deleted scan ID: {scan_id}",
        ip_address=request.remote_addr
    )
    db.add(log)
    db.commit()

    return jsonify({"message": "Scan deleted successfully"}), 200

# --- LOGS ---
@api_bp.route("/logs", methods=["GET"])
def get_logs():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    skip = int(request.args.get("skip", 0))
    limit = int(request.args.get("limit", 50))

    db = SessionLocal()
    logs = db.query(PlagiarismLog).filter(
        PlagiarismLog.user_id == user.id
    ).order_by(PlagiarismLog.created_at.desc()).offset(skip).limit(limit).all()

    return jsonify([{
        "id": l.id,
        "action": l.action,
        "input_type": l.input_type,
        "word_count": l.word_count,
        "result": l.result,
        "similarity_score": l.similarity_score,
        "ip_address": l.ip_address,
        "created_at": l.created_at.isoformat()
    } for l in logs]), 200

# --- ADMIN PANEL ---
@api_bp.route("/admin/stats", methods=["GET"])
def get_admin_stats():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    db = SessionLocal()
    total_users = db.query(User).count()
    total_scans = db.query(DetectionResult).count()
    total_logs = db.query(PlagiarismLog).count()

    avg_score = db.query(DetectionResult.similarity).all()
    avg_similarity = sum([s[0] for s in avg_score]) / len(avg_score) if avg_score else 0

    total_words = db.query(DetectionResult).with_entities(
        DetectionResult.word_count_a + DetectionResult.word_count_b
    ).all()
    total_words = sum([w[0] or 0 for w in total_words])

    return jsonify({
        "total_users": total_users,
        "total_scans": total_scans,
        "total_logs": total_logs,
        "avg_similarity": round(avg_similarity, 4),
        "total_words": total_words
    }), 200

@api_bp.route("/admin/users", methods=["GET"])
def admin_get_users():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    skip = int(request.args.get("skip", 0))
    limit = int(request.args.get("limit", 50))

    db = SessionLocal()
    users = db.query(User).offset(skip).limit(limit).all()

    user_list = []
    for u in users:
        scan_count = db.query(DetectionResult).filter(DetectionResult.user_id == u.id).count()
        user_list.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "is_admin": u.is_admin,
            "scan_count": scan_count,
            "created_at": u.created_at.isoformat()
        })

    return jsonify(user_list), 200

@api_bp.route("/admin/users", methods=["POST"])
def admin_create_user():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    is_admin = data.get("is_admin", False)

    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    db = SessionLocal()
    if db.query(User).filter(User.username == username).first():
        return jsonify({"error": "Username exists"}), 409
    if db.query(User).filter(User.email == email).first():
        return jsonify({"error": "Email exists"}), 409

    new_user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        is_admin=is_admin
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return jsonify({
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "is_admin": new_user.is_admin
    }), 201

@api_bp.route("/admin/users/<int:user_id>", methods=["DELETE"])
def admin_delete_user(user_id):
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    if user_id == user.id:
        return jsonify({"error": "Cannot delete yourself"}), 400

    db = SessionLocal()
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        return jsonify({"error": "User not found"}), 404

    db.query(DetectionResult).filter(DetectionResult.user_id == user_id).delete()
    db.query(PlagiarismLog).filter(PlagiarismLog.user_id == user_id).delete()
    db.delete(target)
    db.commit()

    return jsonify({"message": "User deleted"}), 200

@api_bp.route("/admin/users/<int:user_id>", methods=["PUT"])
def admin_update_user(user_id):
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.json or {}
    db = SessionLocal()
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        return jsonify({"error": "User not found"}), 404

    if "username" in data:
        target.username = data["username"]
    if "email" in data:
        target.email = data["email"]
    if "is_admin" in data:
        target.is_admin = data["is_admin"]
    if "password" in data:
        target.password_hash = generate_password_hash(data["password"])

    db.commit()

    return jsonify({
        "id": target.id,
        "username": target.username,
        "email": target.email,
        "is_admin": target.is_admin
    }), 200

@api_bp.route("/admin/scans", methods=["GET"])
def admin_get_scans():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    skip = int(request.args.get("skip", 0))
    limit = int(request.args.get("limit", 50))

    db = SessionLocal()
    scans = db.query(DetectionResult).order_by(
        DetectionResult.created_at.desc()
    ).offset(skip).limit(limit).all()

    result = []
    for s in scans:
        u = db.query(User).filter(User.id == s.user_id).first()
        result.append({
            "id": s.id,
            "user_id": s.user_id,
            "username": u.username if u else "Unknown",
            "similarity": s.similarity,
            "source_info": s.source_info,
            "input_type": s.input_type,
            "word_count_a": s.word_count_a,
            "word_count_b": s.word_count_b,
            "word_count": (s.word_count_a or 0) + (s.word_count_b or 0),
            "created_at": s.created_at.isoformat()
        })

    return jsonify(result), 200

@api_bp.route("/admin/logs", methods=["GET"])
def admin_get_logs():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    skip = int(request.args.get("skip", 0))
    limit = int(request.args.get("limit", 100))

    db = SessionLocal()
    logs = db.query(PlagiarismLog).order_by(
        PlagiarismLog.created_at.desc()
    ).offset(skip).limit(limit).all()

    result = []
    for l in logs:
        u = db.query(User).filter(User.id == l.user_id).first()
        result.append({
            "id": l.id,
            "user_id": l.user_id,
            "username": u.username if u else "Unknown",
            "action": l.action,
            "input_type": l.input_type,
            "word_count": l.word_count,
            "result": l.result,
            "similarity_score": l.similarity_score,
            "ip_address": l.ip_address,
            "created_at": l.created_at.isoformat()
        })

    return jsonify(result), 200

@api_bp.route("/admin/logs", methods=["DELETE"])
def admin_clear_logs():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    db = SessionLocal()
    db.query(PlagiarismLog).delete()
    db.commit()

    return jsonify({"message": "All logs cleared"}), 200

# --- HEALTH CHECK ---
@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "VeriCheck API running",
        "free_tier_limit": FREE_WORD_LIMIT
    }), 200


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT ROUTES  —  TXT · DOCX · PDF · PNG
# ═══════════════════════════════════════════════════════════════════════════

def _build_report_text(data: dict, title: str = "VeriCheck Plagiarism Report") -> str:
    """Return a plain-text version of any detect result dict."""
    lines = [
        "=" * 70,
        f"  {title}",
        "=" * 70,
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  VeriCheck  : AI-Powered Plagiarism Detector | IIT Patna",
        "-" * 70,
        "",
        "─ RESULT SUMMARY",
        f"  Similarity Score : {data.get('similarity', 0)*100:.1f}%",
        f"  Risk Level       : {data.get('level', 'unknown').upper()}",
        f"  Word Count       : {data.get('word_count', 0)}",
        f"  Character Count  : {data.get('char_count', 0)}",
        "",
    ]

    if data.get("text"):
        lines += [
            "─ INPUT TEXT",
            "  " + "\n  ".join(data["text"].strip().split("\n")),
            "",
        ]

    if "metrics" in data and data["metrics"]:
        m = data["metrics"]
        lines += [
            "─ DETECTION METRICS",
            f"  Sentences Analyzed : {m.get('sentences_analyzed', 'N/A')}",
            f"  Avg Similarity     : {m.get('avg_similarity', 0)*100:.1f}%",
            f"  Max Similarity     : {m.get('max_similarity', 0)*100:.1f}%",
            f"  LSTM Best Score    : {m.get('lstm_best', 0)*100:.1f}%",
            f"  Risky Sentences    : {m.get('risky_sentences', 0)}",
            "",
        ]

    if "sentence_analysis" in data and data["sentence_analysis"]:
        lines += ["─ SENTENCE-LEVEL ANALYSIS", ""]
        for i, s in enumerate(data["sentence_analysis"], 1):
            lines += [
                f"  Sentence {i}: [{s.get('level','?').upper()}] (score: {s.get('score',0)*100:.0f}%)",
                f"    Text : {s.get('sentence','')}",
            ]
            if s.get("match"):
                lines.append(f"    Match: {s.get('match','')[:80]}...")
            if s.get("reason"):
                lines.append(f"    Note : {s.get('reason','')}")
            lines.append("")

    if "explanation" in data:
        lines += ["─ AI EXPLANATION", f"  {data['explanation']}", ""]

    lines += [
        "-" * 70,
        "  VeriCheck — IIT Patna Capstone Project",
        "  Model: Siamese LSTM (62.3% acc) + TF-IDF + N-gram + Char N-gram + LCS",
        "=" * 70,
    ]
    return "\n".join(lines)


# ─── TXT ────────────────────────────────────────────────────────────────────
@api_bp.route("/export/txt", methods=["POST"])
def export_txt():
    """
    POST /api/export/txt
    Body: { text, similarity, level, word_count, char_count,
            metrics, sentence_analysis, explanation }
    Returns: text/plain .txt file download
    """
    payload = request.get_json() or {}
    text = _build_report_text(payload)
    return text, 200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": f"attachment; filename=vericheck-report-{datetime.now():%Y%m%d-%H%M%S}.txt"
    }


# ─── DOCX ───────────────────────────────────────────────────────────────────
@api_bp.route("/export/docx", methods=["POST"])
def export_docx():
    """
    POST /api/export/docx
    Body: same as /export/txt
    Returns: application/vnd.openxmlformats-officedocument.wordprocessingml.document
    """
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from io import BytesIO

    payload = request.get_json() or {}
    doc = DocxDocument()

    # Title
    title = doc.add_heading("VeriCheck Plagiarism Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Meta
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("VeriCheck — AI-Powered Plagiarism Detector | IIT Patna")
    doc.add_paragraph("")

    # Summary section
    doc.add_heading("Result Summary", level=1)
    table = doc.add_table(rows=5, cols=2)
    table.style = "Light Shading Accent 1"
    row_data = [
        ("Similarity Score", f"{payload.get('similarity', 0)*100:.1f}%"),
        ("Risk Level", payload.get('level', 'unknown').upper()),
        ("Word Count", str(payload.get('word_count', 0))),
        ("Character Count", str(payload.get('char_count', 0))),
        ("Explanation", payload.get('explanation', 'N/A')),
    ]
    for i, (k, v) in enumerate(row_data):
        table.rows[i].cells[0].text = k
        table.rows[i].cells[1].text = v

    doc.add_paragraph("")

    # Metrics
    if payload.get("metrics"):
        m = payload["metrics"]
        doc.add_heading("Detection Metrics", level=1)
        m_table = doc.add_table(rows=6, cols=2)
        m_table.style = "Light Shading Accent 1"
        m_data = [
            ("Sentences Analyzed", str(m.get('sentences_analyzed', 'N/A'))),
            ("Avg Similarity", f"{m.get('avg_similarity', 0)*100:.1f}%"),
            ("Max Similarity", f"{m.get('max_similarity', 0)*100:.1f}%"),
            ("LSTM Best Score", f"{m.get('lstm_best', 0)*100:.1f}%"),
            ("Risky Sentences", str(m.get('risky_sentences', 0))),
            ("Model Version", "Siamese LSTM (35%) + TF-IDF (20%) + N-gram (15%) + Char (15%) + LCS (15%)"),
        ]
        for i, (k, v) in enumerate(m_data):
            m_table.rows[i].cells[0].text = k
            m_table.rows[i].cells[1].text = v

    # Sentence analysis
    if payload.get("sentence_analysis"):
        doc.add_paragraph("")
        doc.add_heading("Sentence-Level Analysis", level=1)
        for i, s in enumerate(payload["sentence_analysis"], 1):
            p = doc.add_paragraph()
            run = p.add_run(f"Sentence {i} [{s.get('level','?').upper()}] — Score: {s.get('score',0)*100:.0f}%")
            run.bold = True
            doc.add_paragraph(f'  "{s.get("sentence","")}"')
            if s.get("match"):
                doc.add_paragraph(f"  Match: {s.get('match','')}")
            if s.get("reason"):
                doc.add_paragraph(f"  Note: {s.get('reason','')}")
            doc.add_paragraph("")

    # Explanation
    if payload.get("explanation"):
        doc.add_paragraph("")
        doc.add_heading("AI Explanation", level=1)
        doc.add_paragraph(payload["explanation"])

    # Input text
    if payload.get("text"):
        doc.add_paragraph("")
        doc.add_heading("Input Text", level=1)
        p = doc.add_paragraph(payload["text"])
        p.style = doc.paragraphs[-1].style

    doc.add_paragraph("")
    doc.add_paragraph("─" * 40)
    doc.add_paragraph("VeriCheck — IIT Patna Capstone Project")
    doc.add_paragraph("Model: Ensemble LSTM 62.3% | TF-IDF | N-gram | Char N-gram | LCS")

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    fname = f"vericheck-report-{datetime.now():%Y%m%d-%H%M%S}.docx"
    return buf.read(), 200, {
        "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "Content-Disposition": f"attachment; filename={fname}"
    }


# ─── PDF ─────────────────────────────────────────────────────────────────────
@api_bp.route("/export/pdf", methods=["POST"])
def export_pdf():
    """
    POST /api/export/pdf
    Body: same as /export/txt
    Returns: application/pdf
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import BytesIO

    payload = request.get_json() or {}
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#6366f1"),
                                  spaceAfter=6, alignment=1)  # centered
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
                                     fontSize=10, textColor=colors.grey, alignment=1)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                        textColor=colors.HexColor("#6366f1"), spaceBefore=16, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11,
                        textColor=colors.HexColor("#a855f7"), spaceBefore=10, spaceAfter=4)
    normal = ParagraphStyle("normal", parent=styles["Normal"], fontSize=9, leading=14)
    bold_p = ParagraphStyle("bold", parent=styles["Normal"], fontSize=9, leading=13, fontName="Helvetica-Bold")
    mono = ParagraphStyle("mono", parent=styles["Code"], fontSize=8, leading=12,
                          fontName="Courier", backColor=colors.HexColor("#f5f5f5"))

    story = []

    # Header
    story.append(Paragraph("VeriCheck", title_style))
    story.append(Paragraph("AI-Powered Plagiarism Detection Report", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  IIT Patna Capstone Project",
                            ParagraphStyle("meta", parent=styles["Normal"], fontSize=8,
                                           textColor=colors.grey, alignment=1)))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#6366f1"), spaceAfter=14))

    # Result summary table
    story.append(Paragraph("Result Summary", h1))
    level = payload.get('level', 'unknown').upper()
    level_color = {"SAFE": "#10b981", "LOW": "#84cc16", "MEDIUM": "#f59e0b",
                   "HIGH": "#f97316", "CRITICAL": "#ef4444"}.get(level, "#9ca3af")
    score = payload.get('similarity', 0) * 100

    summary_data = [
        ["Similarity Score", f"{score:.1f}%"],
        ["Risk Level", level],
        ["Word Count", str(payload.get('word_count', 0))],
        ["Character Count", str(payload.get('char_count', 0))],
        ["Explanation", payload.get('explanation', 'N/A')],
    ]
    t = Table(summary_data, colWidths=[4.5*cm, 13*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0ff")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Input text
    if payload.get("text"):
        story.append(Paragraph("Input Text", h1))
        input_para = Paragraph(payload["text"], normal)
        story.append(input_para)
        story.append(Spacer(1, 14))

    # Detection metrics
    if payload.get("metrics"):
        m = payload["metrics"]
        story.append(Paragraph("Detection Metrics", h1))
        m_data = [
            ["Sentences Analyzed", str(m.get('sentences_analyzed', 'N/A'))],
            ["Average Similarity", f"{m.get('avg_similarity', 0)*100:.1f}%"],
            ["Maximum Similarity", f"{m.get('max_similarity', 0)*100:.1f}%"],
            ["LSTM Best Score", f"{m.get('lstm_best', 0)*100:.1f}%"],
            ["Risky Sentences Flagged", str(m.get('risky_sentences', 0))],
        ]
        mt = Table(m_data, colWidths=[5*cm, 12.5*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f5f0ff")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0e0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(mt)
        story.append(Spacer(1, 14))

    # Sentence analysis
    if payload.get("sentence_analysis"):
        story.append(Paragraph("Sentence-Level Analysis", h1))
        for i, s in enumerate(payload["sentence_analysis"], 1):
            level_label = s.get('level', '?').upper()
            level_colors_map = {"SAFE": "#10b981", "LOW": "#84cc16", "MEDIUM": "#f59e0b",
                                "HIGH": "#f97316", "CRITICAL": "#ef4444"}
            lc = level_colors_map.get(level_label, "#9ca3af")
            s_score = s.get('score', 0) * 100

            sent_data = [
                [Paragraph(f"<b>Sentence {i}</b>", normal),
                 Paragraph(f'<font color="{lc}"><b>[{level_label}]</b></font>  Score: {s_score:.0f}%', normal)],
                [Paragraph("Text", bold_p), Paragraph(s.get('sentence', ''), normal)],
            ]
            if s.get("match"):
                sent_data.append([Paragraph("Match", bold_p), Paragraph(s.get('match', ''), normal)])
            if s.get("reason"):
                sent_data.append([Paragraph("Note", bold_p), Paragraph(s.get('reason', ''), normal)])

            st = Table(sent_data, colWidths=[2.5*cm, 15*cm])
            st.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0ff")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0e0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(st)
            story.append(Spacer(1, 6))

    # AI explanation
    if payload.get("explanation"):
        story.append(Spacer(1, 8))
        story.append(Paragraph("AI Explanation", h1))
        story.append(Paragraph(payload["explanation"], normal))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#6366f1")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "VeriCheck — IIT Patna Capstone Project  |  Model: Siamese LSTM (62.3%) + TF-IDF + N-gram + Char N-gram + LCS",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=7.5, textColor=colors.grey, alignment=1)
    ))

    doc.build(story)
    buf.seek(0)
    fname = f"vericheck-report-{datetime.now():%Y%m%d-%H%M%S}.pdf"
    return buf.read(), 200, {
        "Content-Type": "application/pdf",
        "Content-Disposition": f"attachment; filename={fname}"
    }


# ─── PNG (captures the result section as image) ─────────────────────────────
@api_bp.route("/export/png", methods=["POST"])
def export_png():
    """
    POST /api/export/png
    Body: { text, similarity, level, word_count, char_count, metrics, sentence_analysis, explanation }
    Returns: image/png  — renders report as an HTML canvas image
    """
    from io import BytesIO
    import json

    payload = request.get_json() or {}
    similarity = payload.get("similarity", 0)
    level = payload.get("level", "unknown")
    word_count = payload.get("word_count", 0)
    char_count = payload.get("char_count", 0)
    metrics = payload.get("metrics", {})
    sentences = payload.get("sentence_analysis", [])
    explanation = payload.get("explanation", "")

    # SVG → PNG via Cairo (cubicweb does it natively)
    # Fallback: build PNG pixel-by-pixel using reportlab for a styled card image
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, ArcPath, Polygon
        from reportlab.graphics import renderPDF
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.piecharts import Pie

        buf = BytesIO()
        width, height = 800, 1100
        doc = SimpleDocTemplate(buf, pagesize=(width, height),
                                leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0)

        styles = getSampleStyleSheet()

        # Colors
        indigo = colors.HexColor("#6366f1")
        purple = colors.HexColor("#a855f7")
        dark = colors.HexColor("#0f0f1a")
        card_bg = colors.HexColor("#12121f")
        muted = colors.HexColor("#9ca3af")
        white = colors.white

        title_s = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=22,
                                  textColor=indigo, alignment=1, spaceAfter=2)
        sub_s   = ParagraphStyle("s", fontName="Helvetica", fontSize=10,
                                  textColor=muted, alignment=1, spaceAfter=4)
        meta_s  = ParagraphStyle("m", fontName="Helvetica", fontSize=8,
                                  textColor=muted, alignment=1, spaceAfter=2)
        h1_s    = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13,
                                  textColor=indigo, spaceBefore=14, spaceAfter=6)
        cell_s  = ParagraphStyle("c", fontName="Helvetica", fontSize=9,
                                  textColor=white, leading=13)
        bold_s  = ParagraphStyle("b", fontName="Helvetica-Bold", fontSize=9,
                                  textColor=white, leading=13)
        sent_s  = ParagraphStyle("sent", fontName="Helvetica", fontSize=8.5,
                                  textColor=muted, leading=12)

        level_map = {"safe": "#10b981", "low": "#84cc16", "medium": "#f59e0b",
                     "high": "#f97316", "critical": "#ef4444"}
        lc = level_map.get(level.lower(), "#9ca3af")

        story = []

        # Background card
        story.append(Rect(0, 0, width, height, fillColor=dark, strokeColor=None))

        # Title block
        story.append(Paragraph("VeriCheck", title_s))
        story.append(Paragraph("AI Plagiarism Detection Report", sub_s))
        story.append(Paragraph(f"{datetime.now():%Y-%m-%d %H:%M:%S}  •  IIT Patna Capstone", meta_s))

        # Score gauge (text-based)
        score_pct = int(similarity * 100)
        story.append(Spacer(1, 10))
        score_txt = ParagraphStyle("score", fontName="Helvetica-Bold", fontSize=64,
                                    textColor=colors.HexColor(lc), alignment=1)
        story.append(Paragraph(f"{score_pct}%", score_txt))
        lvl_txt = ParagraphStyle("lvl", fontName="Helvetica-Bold", fontSize=16, alignment=1,
                                  textColor=colors.HexColor(lc))
        story.append(Paragraph(f"PLAGIARISM — {level.upper()}", lvl_txt))
        story.append(Spacer(1, 8))

        # Bar chart for methods
        story.append(Paragraph("Detection Breakdown", h1_s))
        method_scores = [
            ("LSTM", metrics.get("lstm_best", 0) * 100, "#6366f1"),
            ("TF-IDF", 0, "#a855f7"),
            ("N-gram", 0, "#10b981"),
            ("Char N-gram", 0, "#f59e0b"),
            ("LCS", 0, "#ef4444"),
        ]
        bar_data = [[s for _, s, _ in method_scores]]
        bar_labels = [l for l, _, _ in method_scores]

        # Stats row
        stats_data = [
            ["Words", "Chars", "Sentences", "Risky"],
            [str(word_count), str(char_count),
             str(metrics.get('sentences_analyzed', '—')),
             str(metrics.get('risky_sentences', 0))],
        ]
        st = Table(stats_data, colWidths=[width/4]*4)
        st.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), indigo),
            ("BACKGROUND", (0, 1), (-1, 1), card_bg),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("TEXTCOLOR", (0, 1), (-1, 1), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2a2a3a")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(st)
        story.append(Spacer(1, 10))

        # Input text
        if payload.get("text"):
            story.append(Paragraph("Input Text", h1_s))
            input_text = payload["text"]
            if len(input_text) > 300:
                input_text = input_text[:300] + "..."
            input_s = ParagraphStyle("inp", fontName="Helvetica", fontSize=9,
                                     textColor=colors.HexColor("#e2e8f0"), leading=13)
            story.append(Paragraph(input_text, input_s))
            story.append(Spacer(1, 10))

        # Sentence analysis
        if sentences:
            story.append(Paragraph("Sentence Analysis", h1_s))
            for s in sentences[:8]:
                sl = level_map.get(s.get('level', 'safe'), "#9ca3af")
                sent_row = [
                    [Paragraph(f'<font color="{sl}">[{s.get("level","?").upper()}]</font> {s.get("sentence","")[:70]}', sent_s)],
                ]
                rt = Table(sent_row, colWidths=[width - 40])
                rt.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#1e1e30")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(rt)
                story.append(Spacer(1, 2))

        # Explanation
        if explanation:
            story.append(Spacer(1, 8))
            story.append(Paragraph("AI Explanation", h1_s))
            story.append(Paragraph(explanation[:300], sent_s))

        # Footer
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width=width, thickness=0.5, color=indigo))
        story.append(Spacer(1, 4))
        footer_s = ParagraphStyle("f", fontName="Helvetica", fontSize=7.5,
                                    textColor=muted, alignment=1)
        story.append(Paragraph(
            "VeriCheck • IIT Patna Capstone Project • Siamese LSTM + Ensemble AI",
            footer_s
        ))

        doc.build(story)
        buf.seek(0)

        return buf.read(), 200, {
            "Content-Type": "image/png",
            "Content-Disposition": f"attachment; filename=vericheck-report-{datetime.now():%Y%m%d-%H%M%S}.png"
        }

    except Exception as e:
        return jsonify({"error": f"PNG generation failed: {str(e)}"}), 500