"""
Curriculum Learning for Nano Quarl
Gradually increase circuit complexity during training
"""
import torch
from typing import List, Dict

class CurriculumScheduler:
    """
    Gradually introduce harder circuits and longer episodes
    """
    def __init__(
        self,
        circuits: List[Dict[str, str]],
        initial_eps_len: int = 50,
        final_eps_len: int = 200,
        warmup_iters: int = 20,
    ):
        # Sort circuits by size (gate count)
        self.circuits = sorted(
            circuits,
            key=lambda c: self._estimate_size(c['qasm'])
        )
        self.initial_eps_len = initial_eps_len
        self.final_eps_len = final_eps_len
        self.warmup_iters = warmup_iters
        
    def _estimate_size(self, qasm: str) -> int:
        """Rough estimate of circuit size"""
        return qasm.count('\n')
    
    def get_curriculum(self, iteration: int) -> Dict:
        """
        Returns current curriculum state
        """
        # Episode length: linear warmup
        if iteration < self.warmup_iters:
            eps_len = int(
                self.initial_eps_len + 
                (self.final_eps_len - self.initial_eps_len) * 
                (iteration / self.warmup_iters)
            )
        else:
            eps_len = self.final_eps_len
        
        # Circuit selection: add one circuit every 10 iterations
        num_circuits = min(
            len(self.circuits),
            1 + iteration // 10
        )
        active_circuits = self.circuits[:num_circuits]
        
        return {
            'eps_len': eps_len,
            'circuits': active_circuits,
            'stage': f'Stage {num_circuits}/{len(self.circuits)}',
        }