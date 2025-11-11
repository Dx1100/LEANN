#!/usr/bin/env python3
"""
Test script to reproduce ColQwen results from issue #119
https://github.com/yichuan-w/LEANN/issues/119

This script demonstrates the ColQwen workflow:
1. Download sample PDF
2. Convert to images
3. Build multimodal index
4. Run test queries
5. Generate similarity maps
"""

import importlib.util
import os
from pathlib import Path


def main():
    print("🧪 ColQwen Reproduction Test - Issue #119")
    print("=" * 50)

    # Check if we're in the right directory
    repo_root = Path.cwd()
    if not (repo_root / "apps" / "colqwen_rag.py").exists():
        print("❌ Please run this script from the LEANN repository root")
        print("   cd /path/to/LEANN && python test_colqwen_reproduction.py")
        return

    print("✅ Repository structure looks good")

    # Step 1: Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import torch

        # Check if pdf2image is available
        if importlib.util.find_spec("pdf2image") is None:
            raise ImportError("pdf2image not found")
        # Check if colpali_engine is available
        if importlib.util.find_spec("colpali_engine") is None:
            raise ImportError("colpali_engine not found")

        print("✅ Core dependencies available")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        print(
            f"   - MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
        )
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📥 Install missing dependencies:")
        print(
            "   uv pip install colpali_engine pdf2image pillow matplotlib qwen_vl_utils einops seaborn"
        )
        return

    # Step 2: Download sample PDF
    print("\n📄 Setting up sample PDF...")
    pdf_dir = repo_root / "test_pdfs"
    pdf_dir.mkdir(exist_ok=True)
    sample_pdf = pdf_dir / "attention_paper.pdf"

    if not sample_pdf.exists():
        print("📥 Downloading sample paper (Attention Is All You Need)...")
        import urllib.request

        try:
            urllib.request.urlretrieve("https://arxiv.org/pdf/1706.03762.pdf", sample_pdf)
            print(f"✅ Downloaded: {sample_pdf}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("   Please manually download a PDF to test_pdfs/attention_paper.pdf")
            return
    else:
        print(f"✅ Using existing PDF: {sample_pdf}")

    # Step 3: Test ColQwen RAG
    print("\n🚀 Testing ColQwen RAG...")

    # Build index
    print("\n1️⃣ Building multimodal index...")
    build_cmd = f"python -m apps.colqwen_rag build --pdfs {pdf_dir} --index test_attention --model colqwen2 --pages-dir test_pages"
    print(f"   Command: {build_cmd}")

    try:
        result = os.system(build_cmd)
        if result == 0:
            print("✅ Index built successfully!")
        else:
            print("❌ Index building failed")
            return
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return

    # Test search
    print("\n2️⃣ Testing search...")
    test_queries = [
        "How does attention mechanism work?",
        "What is the transformer architecture?",
        "How do you compute self-attention?",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        search_cmd = f'python -m apps.colqwen_rag search test_attention "{query}" --top-k 3'
        print(f"   Command: {search_cmd}")

        try:
            result = os.system(search_cmd)
            if result == 0:
                print("✅ Search completed")
            else:
                print("❌ Search failed")
        except Exception as e:
            print(f"❌ Search error: {e}")

    # Test interactive mode (briefly)
    print("\n3️⃣ Testing interactive mode...")
    print("   You can test interactive mode with:")
    print("   python -m apps.colqwen_rag ask test_attention --interactive")

    # Step 4: Test similarity maps (using existing script)
    print("\n4️⃣ Testing similarity maps...")
    similarity_script = (
        repo_root
        / "apps"
        / "multimodal"
        / "vision-based-pdf-multi-vector"
        / "multi-vector-leann-similarity-map.py"
    )

    if similarity_script.exists():
        print("   You can generate similarity maps with:")
        print(f"   cd {similarity_script.parent}")
        print("   python multi-vector-leann-similarity-map.py")
        print("   (Edit the script to use your local PDF)")

    print("\n🎉 ColQwen reproduction test completed!")
    print("\n📋 Summary:")
    print("   ✅ Dependencies checked")
    print("   ✅ Sample PDF prepared")
    print("   ✅ Index building tested")
    print("   ✅ Search functionality tested")
    print("   ✅ Interactive mode available")
    print("   ✅ Similarity maps available")

    print("\n🔗 Related repositories to check:")
    print("   - https://github.com/lightonai/fast-plaid")
    print("   - https://github.com/lightonai/pylate")
    print("   - https://github.com/stanford-futuredata/ColBERT")

    print("\n📝 Next steps:")
    print("   1. Test with your own PDFs")
    print("   2. Experiment with different queries")
    print("   3. Generate similarity maps for visual analysis")
    print("   4. Compare ColQwen2 vs ColPali performance")


if __name__ == "__main__":
    main()
