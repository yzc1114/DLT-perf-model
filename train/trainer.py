from transformers import Trainer


class MTrainer(Trainer):
    def __init__(self, model_init, args, train_dataset, eval_dataset, optimizer_cls):
        super().__init__(model_init=model_init,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset)
        self.model_init = model_init
        self.optimizer_cls = optimizer_cls
        self.args = args

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = model.loss(inputs)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps):
        optimizer = self.optimizer_cls(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer, None
