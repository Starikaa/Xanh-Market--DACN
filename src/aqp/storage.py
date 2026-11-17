from .schemas import Evaluation

class StorageManager:
    def save_evaluation(self, eval_data: Evaluation):
        pass
        
    def export_json(self, eval_data: Evaluation) -> str:
        pass
        
    def export_csv(self, eval_data: Evaluation) -> str:
        pass
