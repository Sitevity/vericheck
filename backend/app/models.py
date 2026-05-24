from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy import create_engine
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SQLITE_DB = os.path.join(BASE_DIR, "..", "plagiarism.db")

engine = create_engine(f"sqlite:///{SQLITE_DB}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    detections = relationship("DetectionResult", back_populates="user", cascade="all, delete-orphan")
    logs = relationship("PlagiarismLog", back_populates="user", cascade="all, delete-orphan")


class DetectionResult(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_text = Column(Text, nullable=False)
    similarity = Column(Float, nullable=False)
    highlighted_html = Column(Text, nullable=False)
    source_info = Column(String(255), nullable=True)
    input_type = Column(String(20), default="text")  # "text" or "file"
    word_count_a = Column(Integer, default=0)
    word_count_b = Column(Integer, default=0)
    chars_a = Column(Integer, default=0)
    chars_b = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="detections")


class PlagiarismLog(Base):
    __tablename__ = "plagiarism_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(50), nullable=False)  # REGISTER, LOGIN, DETECT, DELETE
    input_type = Column(String(20), nullable=True)
    input_chars = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    result = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="logs")