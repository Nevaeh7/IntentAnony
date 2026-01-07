#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt manager usage examples
"""

from prompt_manager import get_manager, get_infer_prompts

def test_basic_usage():
    """Test basic usage"""
    print("=" * 60)
    print("Test 1: Basic usage")
    print("=" * 60)
    
    # Get manager
    manager = get_manager()
    
    # Get Chinese prompt
    zh_prompt = manager.get("infer", "zh")
    print(f"Chinese prompt length: {len(zh_prompt)} characters")
    
    # Get English prompt
    en_prompt = manager.get("infer", "en")
    print(f"English prompt length: {len(en_prompt)} characters")
    
    print("\n Basic usage test passed\n")


def test_list_methods():
    """Test list methods"""
    print("=" * 60)
    print("Test 2: List methods")
    print("=" * 60)
    
    manager = get_manager()
    
    # List all categories
    categories = manager.list_categories()
    print(f"Available categories: {categories}")
    
    # List all languages
    languages = manager.list_languages("infer")
    print(f"Available languages for 'infer' category: {languages}")
    
    print("\n List methods test passed\n")


def test_backward_compatibility():
    """Test backward compatibility"""
    print("=" * 60)
    print("Test 3: Backward compatibility")
    print("=" * 60)
    
    # Use original dictionary method
    infer_prompts = get_infer_prompts()
    
    print(f"Dictionary method - Chinese prompt length: {len(infer_prompts['zh'])} characters")
    print(f"Dictionary method - English prompt length: {len(infer_prompts['en'])} characters")
    
    print("\n Backward compatibility test passed\n")


def test_prompt_content():
    """Test prompt content"""
    print("=" * 60)
    print("Test 4: Prompt content preview")
    print("=" * 60)
    
    manager = get_manager()
    
    # Get first 200 characters of Chinese prompt
    zh_prompt = manager.get("infer", "zh")
    print("Chinese prompt first 200 characters:")
    print(zh_prompt[:200] + "...")
    print()
    
    # Get first 200 characters of English prompt
    en_prompt = manager.get("infer", "en")
    print("English prompt first 200 characters:")
    print(en_prompt[:200] + "...")
    
    print("\n Content preview test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Prompt Manager Test")
    print("=" * 60 + "\n")
    
    try:
        test_basic_usage()
        test_list_methods()
        test_backward_compatibility()
        test_prompt_content()
        
        print("=" * 60)
        print("All tests passed! ")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
