import torch


class st_opt(torch.optim.Optimizer):#Sharpness Tuning Sgd
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(st_opt, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                # 先保存旧参数
                self.state[p]["old_p"] = p.data.clone()

                # 如果没有梯度，跳过更新
                if p.grad is None:
                    continue

                # 计算扰动
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        # 可选：加入微小噪声
        noise_scale = 0.001
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(torch.randn_like(p) * noise_scale)

        if zero_grad:
            self.zero_grad()

    #@torch.no_grad()
    # def first_step(self, zero_grad=False):
    #     grad_norm = self._grad_norm()
    #     for group in self.param_groups:
    #         scale = group["rho"] / (grad_norm + 1e-12)
    #
    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             self.state[p]["old_p"] = p.data.clone()
    #             e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
    #             p.add_(e_w)
    #             #p.add_(torch.randn_like(p) * 0.001)
    #
    #     noise_scale = 0.001  # 微小扰动大小
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             p.add_(torch.randn_like(p) * noise_scale)
    #
    #
    #     if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "optimizer requires a closure function"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    # def _grad_norm(self):
    #     shared_device = self.param_groups[0]["params"][0].device
    #     norm = torch.norm(
    #                 torch.stack([
    #                     ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
    #                     for group in self.param_groups for p in group["params"]
    #                     if p.grad is not None
    #                 ]),
    #                 p=2
    #            )
    #     return norm

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        # 收集有梯度的参数
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    grads.append(grad)

        # 关键修复：如果没有梯度，直接返回 0，不报错
        if len(grads) == 0:
            return torch.tensor(0.0, device=shared_device)

        norm = torch.norm(torch.stack(grads), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
