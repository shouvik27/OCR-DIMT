import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py [opus-mt-en-zh] [finetune|inference]")
        return

    model = sys.argv[1]
    action = sys.argv[2]

    if model == "opus-mt-en-zh":
        if action == "finetune":
            from opus_mt_en_zh.fine_tune.train import start_execution
        elif action == "inference":
            from opus_mt_en_zh.inference.inference import start_execution
        else:
            print("Action must be 'finetune' or 'inference'")
            return
    else:
        print("Model not recognized!")
        return

    start_execution()

if __name__ == "__main__":
    main()
