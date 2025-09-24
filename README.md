# Smart Document Processing API
An intelligent document processing system that automatically extracts entities and validates compliance from contracts and transportation documents using AI and computer vision.

## Project Overview
This API provides smart document processing for two main document types:
* Contracts: Entity extraction using YandexGPT with RAG technology for large documents
* Transportation Documents (TTN): Stamp and signature validation in specific sections using computer vision

## Key Features
For Contracts:
* OCR Processing: Yandex Vision for high-accuracy text recognition
* Entity Extraction: YandexGPT for extracting key entities (shippers, recipients, vehicle numbers, driver phones)
* RAG Technology: Vector search and chunking for large documents
* Violation Detection: AI-powered compliance checking

For Transportation Documents (TTN):
* Document Type Verification: Keyword-based validation 
* Stamp & Signature Detection: Integration with custom YOLO detection API
* Section Analysis: Spatial validation of stamps/signatures in "Cargo Acceptance" section
* Compliance Checking: Ensures required seals and signatures are present

## Technology Stack
* Backend: FastAPI
* OCR: Yandex Vision API
* AI Processing: YandexGPT API
* Computer Vision: Custom YOLO API (previous project)
* RAG: Vector search with keyword matching
* Document Processing: PDF parsing, base64 handling
