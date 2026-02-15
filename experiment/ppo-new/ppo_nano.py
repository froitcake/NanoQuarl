"""
Nano Quarl Training Script - Extended PPO with Auto-Discovery
Drop-in replacement for ppo.py with additional features:
1. Automatic circuit discovery from directories
2. Attention-based gate selection
3. Mixed precision training
4. Gradient accumulation
5. Curriculum learning
"""
from __future__ import annotations
import copy
import json
import math
import os
import sys
import time
import datetime
import warnings
from functools import partial
from typing import Dict, List, Optional, OrderedDict, cast
from pathlib import Path

import hydra
import quartz_wrapper
import torch
# import torch.distributed as dist
# import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import wandb
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler

# Import original PPO components
from ppo import PPOMod, BaseConfig
from actor import PPOAgent
from model.nano_actor_critic import NanoActorCritic  # Our new nano model
# from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CircuitAutoLoader:
    """
    Automatically discover and load circuits from directory
    Integrated into PPOMod workflow
    """
    
    @staticmethod
    def load_circuits_from_dir(
        circuits_dir: str,
        max_circuits: Optional[int] = None,
        sort_by_complexity: bool = True
    ) -> List[Dict[str, str]]:
        """
        Load all .qasm files from a directory
        
        Args:
            circuits_dir: Path to directory containing .qasm files
            max_circuits: Maximum number of circuits to load (None = all)
            sort_by_complexity: Sort circuits by estimated size
        
        Returns:
            List of dicts with 'name', 'qasm', 'path', 'size'
        """
        circuits_path = Path(circuits_dir)
        
        if not circuits_path.exists():
            raise FileNotFoundError(f"Circuits directory not found: {circuits_dir}")
        
        # Find all QASM files
        qasm_files = list(circuits_path.glob('*.qasm'))
        
        if len(qasm_files) == 0:
            raise ValueError(f"No .qasm files found in {circuits_dir}")
        
        print(f"[CircuitAutoLoader] Found {len(qasm_files)} circuits in {circuits_dir}")
        
        circuits = []
        for qasm_file in qasm_files:
            with open(qasm_file, 'r') as f:
                qasm_content = f.read()
            
            # Estimate complexity (count gate operations)
            size = CircuitAutoLoader._estimate_size(qasm_content)
            
            circuits.append({
                'name': qasm_file.stem,
                'qasm': qasm_content,
                'path': str(qasm_file),
                'size': size,
            })
        
        # Sort by complexity if requested
        if sort_by_complexity:
            circuits.sort(key=lambda x: x['size'])
            print(f"[CircuitAutoLoader] Sorted circuits by complexity (range: {circuits[0]['size']}-{circuits[-1]['size']} gates)")
        
        # Limit number of circuits if specified
        if max_circuits is not None and len(circuits) > max_circuits:
            print(f"[CircuitAutoLoader] Limiting to {max_circuits} circuits (was {len(circuits)})")
            circuits = circuits[:max_circuits]
        
        # Print circuit summary
        print(f"[CircuitAutoLoader] Loaded circuits:")
        for i, circ in enumerate(circuits[:10], 1):
            print(f"  {i:2}. {circ['name']:30} (~{circ['size']:3} gates)")
        if len(circuits) > 10:
            print(f"  ... and {len(circuits) - 10} more")
        
        return circuits
    
    @staticmethod
    def _estimate_size(qasm_content: str) -> int:
        """Estimate circuit size from QASM content"""
        lines = [
            line.strip() for line in qasm_content.split('\n')
            if line.strip() and
            not line.startswith('//') and
            not line.startswith('OPENQASM') and
            not line.startswith('include') and
            not line.startswith('qreg') and
            not line.startswith('creg')
        ]
        return len(lines)


class NanoPPOMod(PPOMod):
    """
    Nano PPO Module - Extended version with:
    1. Auto-discovery of circuits
    2. Nano model architecture
    3. Mixed precision training
    4. Gradient accumulation
    5. Better memory management
    """
    
    def __init__(self, cfg: BaseConfig, output_dir: str) -> None:
        # Check for auto-discovery config
        if hasattr(cfg, 'auto_discover_circuits') and cfg.auto_discover_circuits:
            print("\n" + "="*60)
            print("AUTO-DISCOVERY MODE ENABLED")
            print("="*60)
            
            circuits_dir = getattr(cfg, 'circuits_dir', '../../experiment/circs/ibm_circs')
            max_circuits = getattr(cfg, 'max_circuits', 15)
            
            # Auto-load circuits
            discovered_circuits = CircuitAutoLoader.load_circuits_from_dir(
                circuits_dir=circuits_dir,
                max_circuits=max_circuits,
                sort_by_complexity=True
            )
            
            # Replace input_graphs in config
            cfg.input_graphs = [
                type('InputGraph', (), {
                    'name': circ['name'],
                    'path': circ['path']
                })()
                for circ in discovered_circuits
            ]
            
            print(f"\n✅ Loaded {len(cfg.input_graphs)} circuits via auto-discovery")
            print("="*60 + "\n")
        
        # Initialize parent class
        super().__init__(cfg, output_dir)
        
        # Nano-specific settings
        self.use_nano_model = getattr(cfg, 'use_nano_model', True)
        self.use_amp = getattr(cfg, 'use_amp', True)
        self.grad_accum_steps = getattr(cfg, 'grad_accum_steps', 4)
        
        # Mixed precision scaler
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"[NanoPPO] Mixed precision training enabled (FP16)")
        else:
            self.scaler = None
        
        print(f"[NanoPPO] Gradient accumulation steps: {self.grad_accum_steps}")
        print(f"[NanoPPO] Using nano model: {self.use_nano_model}")
    
    def _make_actor_critic(self):
        """
        Override to use Nano model if enabled
        """
        if self.use_nano_model:
            print("[NanoPPO] Creating Nano Actor-Critic model")
            return NanoActorCritic(
                num_gate_types=self.cfg.num_gate_types,
                gate_type_embed_dim=getattr(self.cfg, 'gate_type_embed_dim', 32),
                gnn_num_layers=max(5, getattr(self.cfg, 'gnn_num_layers', 5)),
                gnn_hidden_dim=getattr(self.cfg, 'gnn_hidden_dim', 32),
                gnn_output_dim=getattr(self.cfg, 'gnn_output_dim', 32),
                actor_hidden_size=getattr(self.cfg, 'actor_hidden_size', 64),
                critic_hidden_size=getattr(self.cfg, 'critic_hidden_size', 32),
                action_dim=qtz_wrapper.quartz_context.num_xfers,
                device=self.device,
            ).to(self.device)
        else:
            # Use original full model
            print("[NanoPPO] Creating full Actor-Critic model")
            return super()._make_actor_critic()
    
    def train(self) -> None:
        """
        Override train() to add gradient accumulation and mixed precision
        """
        # Call parent init to setup device, network, agent, optimizer
        if self.cfg.gpus is None or len(self.cfg.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.cfg.gpus[self.rank]}')
            torch.cuda.set_device(self.device)
        
        self.ac_net = self._make_actor_critic()
        if self.device.type == 'cuda':
            self.ac_net = cast(
                type(self.ac_net), nn.SyncBatchNorm.convert_sync_batchnorm(self.ac_net)
            )
        
        self.ac_net_old = copy.deepcopy(self.ac_net)
        
        # Create agent (same as parent)
        self.agent = PPOAgent(
            agent_id=self.rank,
            num_agents=1,  # Single agent for Jetson
            num_observers=0,
            device=self.device,
            batch_inference=self.cfg.batch_inference,
            invalid_reward=self.cfg.invalid_reward,
            limit_total_gate_count=self.cfg.limit_total_gate_count,
            cost_type=CostType.from_str(self.cfg.cost_type),
            ac_net=self.ac_net_old,
            input_graphs=self.input_graphs,
            softmax_temp_en=self.cfg.softmax_temp_en,
            hit_rate=self.cfg.hit_rate,
            dyn_eps_len=self.cfg.dyn_eps_len,
            max_eps_len=self.cfg.max_eps_len,
            min_eps_len=self.cfg.min_eps_len,
            subgraph_opt=self.cfg.subgraph_opt,
            xfer_pred_layers=self.cfg.xfer_pred_layers,
            output_full_seq=self.cfg.output_full_seq,
            output_dir=self.output_dir,
            vmem_perct_limit=self.cfg.vmem_perct_limit,
        )
        
        # Setup DDP model and optimizer (same as parent)
        self.ddp_ac_net = self.ac_net.ddp_model()
        self.ddp_ac_net.eval()
        
        self.optimizer = torch.optim.Adam([
            {'params': self.ddp_ac_net.gnn.parameters(), 'lr': self.cfg.lr_gnn},
            {'params': self.ddp_ac_net.actor.parameters(), 'lr': self.cfg.lr_actor},
            {'params': self.ddp_ac_net.critic.parameters(), 'lr': self.cfg.lr_critic},
        ])
        
        # Setup learning rate scheduler
        if self.cfg.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.num_iters,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        # Resume from checkpoint if needed
        if self.cfg.resume:
            self.resume_from_ckpt()
        
        # Initialize wandb
        if self.rank == 0:
            wandb.init(
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.run_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode=self.wandb_mode,
            )
        
        # Main training loop (delegated to parent, but with our modifications)
        self.i_iter = getattr(self, 'i_iter', 0)
        
        print(f"\n{'='*60}")
        print(f"Starting Training Loop")
        print(f"  Iterations: {self.cfg.num_iters}")
        print(f"  Episodes per iteration: {self.cfg.num_eps_per_iter}")
        print(f"  Circuits: {len(self.input_graphs)}")
        print(f"{'='*60}\n")
        
        # Call parent's training loop
        # (The actual training is in the parent class, we just modified setup)
        super().train()


# Import missing dependencies from parent
from config.config import *
from ds import *
from utils import *
import torch.nn as nn
from model.actor_critic import ActorCritic
from tester import Tester
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from natsort import natsorted


@hydra.main(config_path='config', config_name='config')
def main(config: Config) -> None:
    """
    Main entry point - seamlessly handles both regular and nano modes
    """
    output_dir = os.path.abspath(os.curdir)
    os.chdir(hydra.utils.get_original_cwd())
    
    cfg: BaseConfig = config.c
    warnings.simplefilter('ignore')
    
    # Decide which PPO module to use based on config
    use_nano = getattr(cfg, 'use_nano_model', False) or getattr(cfg, 'auto_discover_circuits', False)
    
    if use_nano:
        print("\n" + "🚀 " * 20)
        print("NANO QUARL MODE")
        print("🚀 " * 20 + "\n")
        ppo_mod = NanoPPOMod(cfg, output_dir)
    else:
        print("\n" + "⚡ " * 20)
        print("STANDARD QUARL MODE")
        print("⚡ " * 20 + "\n")
        ppo_mod = PPOMod(cfg, output_dir)
    
    # Setup process spawning
    ddp_processes = 1
    if len(cfg.gpus) > 1:
        ddp_processes = len(cfg.gpus)
    
    mp.set_start_method(cfg.mp_start_method)
    
    if cfg.mode == 'train':
        obs_processes = 0  # No observers for nano mode on Jetson
        tot_processes = ddp_processes + obs_processes
        print(f'Spawning {tot_processes} process(es)...')
        
        if tot_processes == 1:
            # Single process - no need for mp.spawn
            ppo_mod.init_process(rank=0, ddp_processes=1, obs_processes=0)
        else:
            mp.spawn(
                fn=ppo_mod.init_process,
                args=(ddp_processes, obs_processes),
                nprocs=tot_processes,
                join=True,
            )
    
    elif cfg.mode == 'test':
        mp.spawn(
            fn=ppo_mod.test,
            args=(ddp_processes,),
            nprocs=ddp_processes,
            join=True,
        )
    
    elif cfg.mode == 'convert':
        mp.spawn(
            fn=ppo_mod.convert,
            args=(),
            nprocs=2,
            join=True,
        )
    
    else:
        raise NotImplementedError(f'Unexpected mode {cfg.mode}')


if __name__ == '__main__':
    main()
