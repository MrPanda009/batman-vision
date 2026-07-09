#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import db

SYSTEM_PROMPT = (
    "You are a precise search assistant. Your task is to extract search keywords from the user's query.\n"
    "Requirements:\n"
    "- Extract individual key terms (nouns, adjectives, objects) as separate keywords (e.g., for 'the mug with the blue logo', extract ['mug', 'blue', 'logo'] rather than combining them).\n"
    "- Output ONLY a raw JSON array of strings containing these 2 to 5 search keywords.\n"
    "- Do NOT wrap the JSON in markdown code blocks (such as ```json or ```).\n"
    "- Do NOT output any reasoning, introductory text, or explanations."
)

def format_timestamp(timestamp_val):
    try:
        dt = datetime.fromtimestamp(float(timestamp_val))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(timestamp_val)

def main():
    # Load env variables from .env
    load_dotenv()
    
    # 1. Parse command line arguments or prompt user
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        try:
            query = input("Enter search query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSearch cancelled.")
            sys.exit(0)
            
    if not query:
        print("Error: Search query cannot be empty.", file=sys.stderr)
        sys.exit(1)
        
    print(f"🔍 Recall Query: '{query}'")
    
    # Initialize DB (creates tables/FTS5 indexes if they don't exist)
    db.init_db()
    
    # 2. Setup OpenAI client for build.nvidia.com
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please check your .env file or export it in your environment.", file=sys.stderr)
        sys.exit(1)
        
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    # 3. Call NVIDIA model to extract keywords (disable thinking mode)
    print("🧠 Extracting search keywords using NVIDIA Nemotron...")
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=15.0
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling NVIDIA API: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Parse keywords
    try:
        clean_content = content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()
        
        keywords = json.loads(clean_content)
        if not isinstance(keywords, list):
            raise ValueError("API response did not return a JSON array")
        keywords = [str(kw).strip() for kw in keywords if kw]
    except Exception as e:
        print(f"Error parsing keywords from model response: {e}", file=sys.stderr)
        print(f"Raw API response:\n{content}", file=sys.stderr)
        sys.exit(1)
        
    if not keywords:
        print("No search keywords could be extracted from your query.")
        sys.exit(0)
        
    print(f"🔑 Extracted Keywords: {keywords}")
    
    # 5. Run keywords through FTS5 OR search
    fts_query = " OR ".join([f'"{kw}"' for kw in keywords])
    print(f"🔎 Querying database with FTS5: MATCH '{fts_query}'...")
    
    try:
        results = db.search_objects(fts_query)
    except Exception as e:
        print(f"Database search failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not results:
        print("\n❌ No matching objects found in the database.")
        sys.exit(0)
        
    # 6. Rank results based on Python-level keyword match count (case-insensitive check)
    ranked_results = []
    for obj in results:
        match_count = 0
        obj_tags = obj.get("tags") or []
        ocr = obj.get("ocr_text") or ""
        
        # Check each keyword
        for kw in keywords:
            kw_lower = kw.lower()
            tag_match = any(kw_lower in tag.lower() for tag in obj_tags)
            ocr_match = kw_lower in ocr.lower()
            
            if tag_match or ocr_match:
                match_count += 1
                
        # Only retain objects that matched at least one keyword in python checking
        if match_count > 0:
            ranked_results.append((obj, match_count))
            
    # Sort by match_count in descending order
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    if not ranked_results:
        print("\n❌ No matching objects found after ranking filter.")
        sys.exit(0)
        
    # 7. Print ranked results
    total_keywords = len(keywords)
    print(f"\n✨ Found {len(ranked_results)} matching object(s) (ranked by relevance):\n")
    
    for rank, (obj, count) in enumerate(ranked_results, start=1):
        print(f"{rank}. [Relevance: {count}/{total_keywords} keywords matched] — Track ID: {obj['track_id']}")
        print(f"   ⏱️  First Seen: {format_timestamp(obj['first_seen'])}")
        print(f"   ⏱️  Last Seen:  {format_timestamp(obj['last_seen'])}")
        
        # Format tags
        tags = obj.get("tags")
        if isinstance(tags, list):
            tags_str = ", ".join(tags)
        else:
            tags_str = str(tags) if tags else "None"
        print(f"   🏷️  Tags:       {tags_str}")
        
        # Format OCR
        ocr_str = obj.get("ocr_text") or "None"
        print(f"   📝 OCR Text:   {ocr_str}")
        
        # Format Crop paths
        crop_paths = obj.get("crop_paths") or []
        print("   🖼️  Crop Paths:")
        for path in crop_paths:
            print(f"     - {path}")
            
        print("=" * 60)

if __name__ == "__main__":
    main()
