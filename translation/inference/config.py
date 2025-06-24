# Configuration settings for the translation process

class TranslationConfig:
    def __init__(self):
        self.model_name = "default_model"
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.max_input_length = 512
        self.max_output_length = 512
        self.use_gpu = True
        self.checkpoint_path = "./checkpoints/"
        self.logging_dir = "./logs/"
        self.data_path = "./data/"
        self.output_path = "./output/"
        
    def display_config(self):
        print("Translation Configuration:")
        print(f"Model Name: {self.model_name}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Max Input Length: {self.max_input_length}")
        print(f"Max Output Length: {self.max_output_length}")
        print(f"Use GPU: {self.use_gpu}")
        print(f"Checkpoint Path: {self.checkpoint_path}")
        print(f"Logging Directory: {self.logging_dir}")
        print(f"Data Path: {self.data_path}")
        print(f"Output Path: {self.output_path}")