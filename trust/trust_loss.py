import torch
import torch.nn as nn
import torch.nn.functional as F


def deactivate_track_running_status(model):
    for name, mm in model.named_modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.track_running_stats = False


def activate_track_running_status(model):
    for name, mm in model.named_modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.track_running_stats = True


@torch.no_grad()
def select_uplow(nat_logit, inv_logit, adv_logit, y, batch_size):
    arange_bs = torch.arange(batch_size, device=nat_logit.device)
    p_nat = nat_logit.softmax(1)[arange_bs, y]
    p_inv = inv_logit.softmax(1)[arange_bs, y]
    p_adv = adv_logit.softmax(1)[arange_bs, y]
    ps = torch.stack([p_nat, p_inv, p_adv])
    u_inds = ps.argmin(dim=0)
    v_inds = ps.argmax(dim=0)

    return u_inds, v_inds


def get_u_v(nat, inv, adv, u_inds, v_inds):
    device = nat.device
    u = torch.zeros(nat.size(), device=device)
    u[u_inds == 0] = nat[u_inds == 0]
    u[u_inds == 1] = inv[u_inds == 1]
    u[u_inds == 2] = adv[u_inds == 2]

    v = torch.zeros(nat.size(), device=device)
    v[v_inds == 0] = nat[v_inds == 0]
    v[v_inds == 1] = inv[v_inds == 1]
    v[v_inds == 2] = adv[v_inds == 2]

    return u, v

def symmkl(u_logit, v_logit):
    log_u = F.log_softmax(u_logit, dim=1)
    log_v = F.log_softmax(v_logit, dim=1)

    u = log_u.exp()
    v = log_v.exp()

    loss = 0.5 * (
        F.kl_div(log_u, u, reduction="batchmean") +
        F.kl_div(log_v, v, reduction="batchmean")
    )
    return loss

def tri_loss(
    model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, loss_mode
):
    batch_size = len(x_natural)
    device = x_natural.device
    model.train()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
    x_inv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
    x_nat = x_natural.clone()
    deactivate_track_running_status(model)
    with torch.no_grad():
        nat_logit = model(x_natural)
    for att_ind in range(perturb_steps):
        nat_logit = nat_logit.data
        x_inv.requires_grad_()
        x_adv.requires_grad_()
        inv_logit = model(x_inv)
        adv_logit = model(x_adv)
        u_inds, v_inds = select_uplow(
            nat_logit, inv_logit, adv_logit, y, batch_size
        )
        nat_need = torch.logical_or(u_inds == 0, v_inds == 0)
        if nat_need.sum() != 0:
            x_nat.requires_grad_()
            nat_logit = model(x_nat)
        u_logit, v_logit = get_u_v(
            nat_logit, inv_logit, adv_logit, u_inds, v_inds
        )

        loss_att = symmkl(u_logit, v_logit)

        if nat_need.sum() != 0:
            grad = torch.autograd.grad(
                loss_att, [x_nat, x_inv, x_adv], allow_unused=True
            )
            x_adv, x_inv = get_u_v(
                x_natural.data, x_inv.data, x_adv.data, u_inds, v_inds
            )
            grad_adv, grad_inv = get_u_v(
                grad[0], grad[1], grad[2], u_inds, v_inds
            )
        else:
            grad = torch.autograd.grad(loss_att, [x_inv, x_adv], allow_unused=True)
            grad_adv, grad_inv = get_u_v(
                torch.zeros(x_natural.size(), device=device),
                grad[0],
                grad[1],
                u_inds,
                v_inds,
            )

        x_inv = x_inv.detach() + step_size * torch.sign(grad_inv.detach())
        x_inv = torch.min(torch.max(x_inv, x_natural - epsilon), x_natural + epsilon)
        x_inv = torch.clamp(x_inv, 0.0, 1.0)
        x_adv = x_adv.detach() + step_size * torch.sign(grad_adv.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    inv_logit = model(x_inv)
    activate_track_running_status(model)
    nat_logit = model(x_natural)
    adv_logit = model(x_adv)

    u_inds, v_inds = select_uplow(
        nat_logit, inv_logit, adv_logit, y, batch_size
    )
    u_logit, v_logit = get_u_v(
        nat_logit, inv_logit, adv_logit, u_inds, v_inds
    )

    loss_natural = F.cross_entropy(nat_logit, y)
    loss_robust = symmkl(u_logit, v_logit)
    loss = loss_natural + beta * loss_robust

    with torch.no_grad():
        ce_u = F.cross_entropy(u_logit, y)
        ce_v = F.cross_entropy(v_logit, y)
        ce_clean = F.cross_entropy(nat_logit, y)
    return loss, ce_u, ce_clean, ce_v


def trust_loss(
    model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, loss_mode
):
    batch_size = len(x_natural)
    device = x_natural.device
    model.train()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
    x_inv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
    x_nat = x_natural.clone()
    deactivate_track_running_status(model)
    with torch.no_grad():
        nat_logit = model(x_natural)
    for att_ind in range(perturb_steps):
        nat_logit = nat_logit.data
        x_inv.requires_grad_()
        x_adv.requires_grad_()
        inv_logit = model(x_inv)
        adv_logit = model(x_adv)
        u_inds, v_inds = select_uplow(
            nat_logit, inv_logit, adv_logit, y, batch_size
        )

        nat_need = torch.logical_or(u_inds == 0, v_inds == 0)
        if nat_need.sum() != 0:
            x_nat.requires_grad_()
            nat_logit = model(x_nat)
        u_logit, v_logit = get_u_v(
            nat_logit, inv_logit, adv_logit, u_inds, v_inds
        )

        loss_att = symmkl(u_logit, v_logit)
        if nat_need.sum() != 0:
            grad = torch.autograd.grad(
                loss_att, [x_nat, x_inv, x_adv], allow_unused=True
            )
            x_adv, x_inv = get_u_v(
                x_natural.data, x_inv.data, x_adv.data, u_inds, v_inds
            )
            grad_adv, grad_inv = get_u_v(
                grad[0], grad[1], grad[2], u_inds, v_inds
            )
        else:
            grad = torch.autograd.grad(loss_att, [x_inv, x_adv], allow_unused=True)
            grad_adv, grad_inv = get_u_v(
                torch.zeros(x_natural.size(), device=device),
                grad[0],
                grad[1],
                u_inds,
                v_inds,
            )
        x_inv = x_inv.detach() - step_size * torch.sign(grad_inv.detach())
        x_inv = torch.min(torch.max(x_inv, x_natural - epsilon), x_natural + epsilon)
        x_inv = torch.clamp(x_inv, 0.0, 1.0)
        x_adv = x_adv.detach() + step_size * torch.sign(grad_adv.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv, x_inv