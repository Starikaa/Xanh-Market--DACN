from .schemas import Evaluation

class EvaluationPipeline:
    def __init__(self):
        pass

    def run(self, article: str, evidence: list) -> Evaluation:
        """Điều phối các module: Nạp -> Tách claim -> Đối chiếu -> Chẩn đoán -> Gợi ý -> Xuất"""
        pass
