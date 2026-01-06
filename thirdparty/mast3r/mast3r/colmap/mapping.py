# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R-SLAM mapping utilities
# --------------------------------------------------------
import numpy as np
import torch
from mast3r.cloud_opt.base_opt import BasePCOptimizer


class Mapping:
    def __init__(self, shared_intrinsics=True, same_focals=True):
        self.shared_intrinsics = shared_intrinsics
        self.same_focals = same_focals
        self.optimizer = None
        
    def init_optimizer(self, view1, view2, pred1, pred2):
        """Initialize the optimizer with two views."""
        self.optimizer = BasePCOptimizer(
            [view1, view2],
            [pred1, pred2],
            shared_intrinsics=self.shared_intrinsics,
            same_focals=self.same_focals
        )
        
    def add_view(self, view, pred):
        """Add a new view to the optimization."""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Call init_optimizer first.")
        self.optimizer.add_view(view, pred)
        
    def optimize(self, n_iters=300):
        """Run optimization."""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Call init_optimizer first.")
        return self.optimizer.optimize(n_iters=n_iters)
        
    def get_pointcloud(self):
        """Get the optimized point cloud."""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized.")
        return self.optimizer.get_dense_pts3d()
        
    def get_cameras(self):
        """Get the optimized camera parameters."""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized.")
        return self.optimizer.get_cameras()
