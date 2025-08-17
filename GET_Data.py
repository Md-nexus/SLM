from datasets import load_dataset
import json

def get_daily_dialog():
    """Get DailyDialog dataset"""
    print("Loading DailyDialog dataset...")
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    return dataset

def get_openassistant():
    """Get OpenAssistant dataset for more diverse conversations"""
    print("Loading OpenAssistant dataset...")
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        return dataset
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")
        return None

def format_daily_dialog(dataset):
    """Format DailyDialog conversations"""
    formatted_data = []
    for dialog in dataset["train"]:
        turns = dialog["dialog"]
        for i in range(len(turns) - 1):
            q = turns[i].strip().lower()
            a = turns[i + 1].strip().lower()
            if q and a and len(q) > 5 and len(a) > 5:  # Filter out very short exchanges
                formatted_data.append(f"Q: {q}\nA: {a}\n===\n")
    return formatted_data

def format_openassistant(dataset):
    """Format OpenAssistant conversations"""
    formatted_data = []
    if dataset is None:
        return formatted_data
    
    # Group by conversation
    conversations = {}
    for item in dataset:
        conv_id = item.get('message_tree_id', item.get('id', 'unknown'))
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(item)
    
    # Sort each conversation by timestamp or message order
    for conv_id, messages in conversations.items():
        messages.sort(key=lambda x: x.get('created_date', 0))
        
        # Extract Q&A pairs
        for i in range(len(messages) - 1):
            msg1 = messages[i]
            msg2 = messages[i + 1]
            
            if msg1.get('role') == 'user' and msg2.get('role') == 'assistant':
                q = msg1.get('text', '').strip().lower()
                a = msg2.get('text', '').strip().lower()
                
                if q and a and len(q) > 5 and len(a) > 5:
                    formatted_data.append(f"Q: {q}\nA: {a}\n===\n")
    
    return formatted_data

def main():
    """Main function to collect and combine datasets"""
    print("🚀 Collecting enhanced dataset...")
    
    # Get datasets
    daily_dialog = get_daily_dialog()
    openassistant = get_openassistant()
    
    # Format datasets
    daily_data = format_daily_dialog(daily_dialog)
    oa_data = format_openassistant(openassistant)
    
    # Combine datasets
    all_data = daily_data + oa_data
    
    # Shuffle the data
    import random
    random.shuffle(all_data)
    
    # Save to file
    output_path = "data/brain.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(all_data)
    
    print(f"✅ Dataset saved to {output_path}")
    print(f"📊 Total conversations: {len(all_data)}")
    print(f"   - DailyDialog: {len(daily_data)}")
    print(f"   - OpenAssistant: {len(oa_data)}")
    
    # Show some examples
    print("\n📝 Sample conversations:")
    for i, conv in enumerate(all_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(conv.strip())

if __name__ == "__main__":
    main()
