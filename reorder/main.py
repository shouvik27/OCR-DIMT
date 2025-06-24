import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py [LayoutLMv3_T5|LayoutLMv3_T2] [finetune|inference]")
        return

    model = sys.argv[1]
    action = sys.argv[2]

    if model == "LayoutLMv3_T5":
        if action == "finetune":
            from reorder.LayoutLMv3_T5.fine_tune.train import start_execution
        elif action == "inference":
            from reorder.LayoutLMv3_T5.inference import start_execution
        else:
            print("Action must be 'finetune' or 'inference'")
            return
    elif model == "Llama_4_Maverick":
        if action == "finetune":
            from Llama_4_Maverick.fine_tune.train import start_execution
        elif action == "inference":
            from Llama_4_Maverick.inference.inference import start_execution
        else:
            print("Action must be 'finetune' or 'inference'")
            return
    elif model == "Pixtral_12B":
        if action == "finetune":
            from Pixtral.fine_tune.train import start_execution
        elif action == "inference":
            from Pixtral.inference.inference import start_execution    
        else:
            print("Action must be 'finetune' or 'inference'")
            return    
    else:
        print("Model not recognized!")
        return

    start_execution()

if __name__ == "__main__":
    main()
