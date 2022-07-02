from copy import deepcopy
import torch
from torch import nn
from tqdm.autonotebook import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from harmonic_loss import harmonic_loss
from util import *

def simple_pretrain_loop(
            model, pretrain_loader, opt, n_epochs=1, loss_fn=nn.L1Loss(), lr_sched=None,
        ):
    device = model.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
    model["trans"] = model["trans"].to(device)
    model["trans"].train()
    for epoch_n in tqdm(range(1, n_epochs+1)):
        print("epoch", epoch_n)
        torch.cuda.empty_cache()
        for batch in tqdm(pretrain_loader):
            opt.zero_grad()
            batch = batch.to(device)
            transformed = model["trans"](batch)
            loss = loss_fn(transformed, batch)
            loss.backward()
            opt.step()
        if lr_sched is not None:
            lr_sched.step()
    
    for batch in pretrain_loader:
        pics = batch[:5].to(device)
        tr = model["trans"](pics)
        model["trans"].eval()
        ev = model["trans"](pics)
        model["trans"].train()
        draw(pics)
        draw(tr)
        draw(ev)
        plt.show()
        break

def simple_train_loop(
        model, photo_loader, icon_loader, test_loader, n_epochs, loss_fn, opt, lr_sched=None,
        d_loss_boundary=1.0, t_loss_boundary=5.0, max_iter=5,
        lbl_noise=0.05, buffer=3,
    ):
    device = model.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
    model["trans"] = model["trans"].to(device)
    model["disc"] = model["disc"].to(device)
    model["trans"].train()
    model["disc"].train()
    buffer = DiscriminationBuffer(buffer)

    disc_losses = []
    trans_losses = []

    for epoch_n in tqdm(range(1, n_epochs+1)):
        print("epoch", epoch_n)
        torch.cuda.empty_cache()

        disc_epoch_losses = []
        trans_epoch_losses = []
        real_epoch_scores = []
        fake_epoch_scores = []
        photo_epoch_scores = []

        buffer.put(model["trans"].state_dict()) # no deepcopy here
        for icon_batch in tqdm(icon_loader):
            disc_batch_losses, trans_batch_losses, real_batch_scores, fake_batch_scores, photo_batch_scores = simple_train_batch(
                    model, buffer, photo_loader, icon_batch, loss_fn, opt,
                    d_loss_boundary, t_loss_boundary, max_iter, lbl_noise,
                    device=device,
                )
            disc_epoch_losses += disc_batch_losses
            trans_epoch_losses += trans_batch_losses
            real_epoch_scores += real_batch_scores
            fake_epoch_scores += fake_batch_scores
            photo_epoch_scores += photo_batch_scores

        buffer.items[-1] = deepcopy(model["trans"].state_dict())

        epochal_disc_loss = np.mean(disc_epoch_losses)
        disc_losses.append(epochal_disc_loss)
        epochal_trans_loss = np.mean(trans_epoch_losses)
        trans_losses.append(epochal_trans_loss)
        print(f"disc loss: {epochal_disc_loss:.4f}, trans loss: {epochal_trans_loss:.4f}")
        epochal_real_score = np.mean(real_epoch_scores)
        epochal_fake_score = np.mean(fake_epoch_scores)
        epochal_photo_score = np.mean(photo_epoch_scores)
        print(f"real score: {epochal_real_score:.4f}, fake score: {epochal_fake_score:.4f}, photo score: {epochal_photo_score:.4f}")

        if lr_sched is not None:
            if "disc" in lr_sched:
                lr_sched["disc"].step()
            if "trans" in lr_sched:
                lr_sched["trans"].step()

        simple_draw_interm(model, test_loader)
        plt.show()

def simple_train_batch(
            model, buffer, photo_loader, icon_batch, loss_fn, opt,
            d_loss_boundary, t_loss_boundary, max_iter, lbl_noise,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
    batch_size = icon_batch.shape[0]
    icon_batch = icon_batch.to(device)
    disc_batch_losses = []
    trans_batch_losses = []
    real_batch_scores = []
    fake_batch_scores = []
    photo_batch_scores = []
    # train discriminator
    disc_loss = d_loss_boundary
    iter = 0
    while disc_loss >= d_loss_boundary and iter < max_iter:
        iter += 1
        opt["disc"].zero_grad()
        # icons
        # icon_target = torch.ones(batch_size, 1, device=device)
        icon_target = torch.from_numpy(np.random.uniform(0, lbl_noise, size=(batch_size,1),)) # label flip and noize
        icon_target = icon_target.float().to(device)
        icon_pred = model["disc"](icon_batch)
        icon_loss = loss_fn["disc_real"](icon_pred, icon_target)
        # real_batch_scores.append(torch.mean(icon_pred).item())
        real_batch_scores.append(1 - torch.mean(icon_pred).item()) # label flip
        # fakes
        photo_batch = photo_loader.get_batch(batch_size).to(device)
        model["trans"].load_state_dict(buffer.get())
        # print(model["trans"])
        # print(fake_batch)
        fake_batch = model["trans"](photo_batch)
        # model["trans"].load(buffer.get(-1))
        # fake_target = torch.zeros(batch_size, 1, device=device)
        fake_target = torch.from_numpy(np.random.uniform(1, 1-lbl_noise, size=(batch_size,1),)) # label flip and noize
        fake_target = fake_target.float().to(device)
        fake_pred = model["disc"](fake_batch)
        fake_loss = loss_fn["disc_fake"](fake_pred, fake_target)
        # backprop
        disc_loss = icon_loss + fake_loss
        disc_loss.backward()
        opt["disc"].step()
        disc_batch_losses.append(disc_loss.item())
        # fake_batch_scores.append(torch.mean(fake_pred).item())
        fake_batch_scores.append(1 - torch.mean(fake_pred).item()) # label flip
        photo_pred = model["disc"](photo_batch)
        photo_batch_scores.append(1 - torch.mean(photo_pred).item()) # label flip

    model["trans"].load_state_dict(buffer.get(-1))

    # train transformator
    trans_loss = t_loss_boundary
    iter = 0
    while trans_loss >= t_loss_boundary and iter < max_iter:
        iter += 1
        opt["trans"].zero_grad()
        # trans_target = torch.ones(batch_size, 1, device=device)
        trans_target = torch.from_numpy(np.random.uniform(0, lbl_noise, size=(batch_size,1),)) # label flip and noize
        trans_target = trans_target.float().to(device)
        fake_batch = photo_loader.get_batch(batch_size).to(device)
        fake_batch = model["trans"](fake_batch)
        trans_pred = model["disc"](fake_batch)
        trans_loss = loss_fn["trans"](trans_pred, trans_target)
        trans_loss.backward()
        opt["trans"].zero_grad()
        trans_batch_losses.append(trans_loss.item())

    return disc_batch_losses, trans_batch_losses, real_batch_scores, fake_batch_scores, photo_batch_scores

def simple_draw_interm(model, test_loader, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    test_batch = test_loader.get_batch(3).to(device)
    train_fakes = model["trans"](test_batch)
    model["trans"].eval()
    eval_fakes = model["trans"](test_batch)
    model["trans"].train()
    # train_fakes = train_fakes.detach().cpu().numpy()
    # eval_fakes = eval_fakes.detach().cpu().numpy()
    fakes = torch.concat((test_batch, train_fakes, eval_fakes))
    # fig, axes = plt.subplots(1, 6, sharey=True, figsize=(12,3))
    titles = [""] + ["orig"] + 2*[""] + ["train"] + [""]*2 + ["eval"] + [""]
    draw(fakes, titles)
    # for i, fake in enumerate(fakes):
    #     axes[i].imshow(np.rollaxis(fake, 0, 3))
    #     axes[i]('off')
    #     axes[i].title(titles[i])

def draw(pics, titles=None):
    count = pics.shape[0]
    fig, axes = plt.subplots(1, count, sharey=True, figsize=(3*count,3))
    # titles = [""] + ["train"] + [""]*2 + ["eval"] + [""]
    for i, pic in enumerate(pics):
        pic = pic.detach().cpu().squeeze().numpy()
        axes[i].imshow(np.rollaxis(pic, 0, 3))
        axes[i].axis('off')
        if titles is not None:
          axes[i].set_title(titles[i])


def cycle_train_loop(
            forward_model, backward_model,
            forward_photo_loader, forward_icon_loader, forward_test_loader,
            backward_photo_loader, backward_icon_loader, backward_test_loader,
            n_epochs, forward_loss_fn, forward_opt, backward_loss_fn, backward_opt,
            cycle_opt, cycle_coef, cycle_loss_fn=nn.MSELoss(), lr_sched=None,
            forward_d_loss_boundary=1.0, forward_t_loss_boundary=5.0,
            backward_d_loss_boundary = 1.0, backward_t_loss_boundary=5.0,
            max_iter=3,
            lbl_noise=0.05, buffer=3,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
    forward_model["trans"] = forward_model["trans"].to(device)
    forward_model["disc"] = forward_model["disc"].to(device)
    forward_model["trans"].train()
    forward_model["disc"].train()
    backward_model["trans"] = backward_model["trans"].to(device)
    backward_model["disc"] = backward_model["disc"].to(device)
    backward_model["trans"].train()
    backward_model["disc"].train()
    forward_buffer = DiscriminationBuffer(buffer)
    backward_buffer = DiscriminationBuffer(buffer)

    forward_disc_losses = []
    forward_trans_losses = []
    backward_disc_losses = []
    backward_trans_losses = []

    for epoch_n in tqdm(range(1, n_epochs+1)):
        print("epoch", epoch_n)
        torch.cuda.empty_cache()

        forward_disc_epoch_losses = []
        forward_trans_epoch_losses = []
        forward_real_epoch_scores = []
        forward_fake_epoch_scores = []
        forward_photo_epoch_scores = []
        backward_disc_epoch_losses = []
        backward_trans_epoch_losses = []
        backward_real_epoch_scores = []
        backward_fake_epoch_scores = []
        backward_icon_epoch_scores = []
        cycle_consistency_losses = []

        forward_buffer.put(forward_model["trans"].state_dict()) # no deepcopy here
        backward_buffer.put(forward_model["trans"].state_dict()) # no deepcopy here
        for forward_icon_batch, backward_photo_batch in tqdm(zip(forward_icon_loader, backward_photo_loader)):
            forward_icon_batch, backward_photo_batch = forward_icon_batch.to(device), backward_photo_batch.to(device)
            disc_batch_losses, trans_batch_losses, real_batch_scores, fake_batch_scores, photo_batch_scores = simple_train_batch(
                    forward_model, forward_buffer, forward_photo_loader, forward_icon_batch, forward_loss_fn, forward_opt,
                    forward_d_loss_boundary, forward_t_loss_boundary, max_iter, lbl_noise,
                    device=device,
                )
            forward_disc_epoch_losses += disc_batch_losses
            forward_trans_epoch_losses += trans_batch_losses
            forward_real_epoch_scores += real_batch_scores
            forward_fake_epoch_scores += fake_batch_scores
            forward_photo_epoch_scores += photo_batch_scores

            disc_batch_losses, trans_batch_losses, real_batch_scores, fake_batch_scores, icon_batch_scores = simple_train_batch(
                    backward_model, backward_buffer, backward_icon_loader, backward_photo_batch, backward_loss_fn, backward_opt,
                    backward_d_loss_boundary, backward_t_loss_boundary, max_iter, lbl_noise,
                    device=device,
                )
            backward_disc_epoch_losses += disc_batch_losses
            backward_trans_epoch_losses += trans_batch_losses
            backward_real_epoch_scores += real_batch_scores
            backward_fake_epoch_scores += fake_batch_scores
            backward_icon_epoch_scores += icon_batch_scores

            cycle_opt.zero_grad()
            forward_doppler = backward_model["trans"](forward_model["trans"](backward_photo_batch)) # nevermind forward-backward mixing
            forward_cycle_loss = cycle_coef*cycle_loss_fn(backward_photo_batch, forward_doppler)
            backward_doppler = forward_model["trans"](backward_model["trans"](forward_icon_batch))
            backward_cycle_loss = cycle_coef*cycle_loss_fn(forward_icon_batch, backward_doppler)
            cycle_loss = forward_cycle_loss + backward_cycle_loss
            cycle_loss.backward()
            cycle_opt.step()
            cycle_consistency_losses.append(cycle_loss.item())

        forward_buffer.items[-1] = deepcopy(forward_model["trans"].state_dict())
        backward_buffer.items[-1] = deepcopy(backward_model["trans"].state_dict())

        forward_epochal_disc_loss = np.mean(forward_disc_epoch_losses)
        forward_disc_losses.append(forward_epochal_disc_loss)
        forward_epochal_trans_loss = np.mean(forward_trans_epoch_losses)
        forward_trans_losses.append(forward_epochal_trans_loss)
        print(f"forward disc loss: {forward_epochal_disc_loss:.4f}, forward trans loss: {forward_epochal_trans_loss:.4f}")
        forward_epochal_real_score = np.mean(forward_real_epoch_scores)
        forward_epochal_fake_score = np.mean(forward_fake_epoch_scores)
        forward_epochal_photo_score = np.mean(forward_photo_epoch_scores)
        print(f"forward real score: {forward_epochal_real_score:.4f}, forward fake score: {forward_epochal_fake_score:.4f}, forward photo score: {forward_epochal_photo_score:.4f}")
        backward_epochal_disc_loss = np.mean(backward_disc_epoch_losses)
        backward_disc_losses.append(backward_epochal_disc_loss)
        backward_epochal_trans_loss = np.mean(backward_trans_epoch_losses)
        backward_trans_losses.append(backward_epochal_trans_loss)
        print(f"backward disc loss: {backward_epochal_disc_loss:.4f}, backward trans loss: {backward_epochal_trans_loss:.4f}")
        backward_epochal_real_score = np.mean(backward_real_epoch_scores)
        backward_epochal_fake_score = np.mean(backward_fake_epoch_scores)
        backward_epochal_icon_score = np.mean(backward_icon_epoch_scores)
        print(f"backward real score: {backward_epochal_real_score:.4f}, backward fake score: {backward_epochal_fake_score:.4f}, backward icon score: {backward_epochal_icon_score:.4f}")
        print(f"cycle consistency loss: {np.mean(cycle_consistency_losses):.4f}")

        if lr_sched is not None:
            if "disc" in lr_sched:
                lr_sched["disc"].step()
            if "trans" in lr_sched:
                lr_sched["trans"].step()

        print("Forward test")
        simple_draw_interm(forward_model, forward_test_loader)
        plt.show()
        print("Backward test")
        simple_draw_interm(backward_model, backward_test_loader)
        plt.show()

def harmonic_train_loop(
            forward_model, backward_model,
            forward_photo_loader, forward_icon_loader, forward_test_loader,
            backward_photo_loader, backward_icon_loader, backward_test_loader,
            n_epochs, forward_loss_fn, forward_opt, backward_loss_fn, backward_opt,
            cycle_opt, cycle_coef, dist_fn=nn.MSELoss(), sigma=0.015625, lr_sched=None,
            forward_d_loss_boundary=1.0, forward_t_loss_boundary=5.0,
            backward_d_loss_boundary = 1.0, backward_t_loss_boundary=5.0,
            max_iter=3,
            lbl_noise=0.05, buffer=3,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
        ):
    forward_model["trans"] = forward_model["trans"].to(device)
    forward_model["disc"] = forward_model["disc"].to(device)
    forward_model["trans"].train()
    forward_model["disc"].train()
    backward_model["trans"] = backward_model["trans"].to(device)
    backward_model["disc"] = backward_model["disc"].to(device)
    backward_model["trans"].train()
    backward_model["disc"].train()
    forward_buffer = DiscriminationBuffer(buffer)
    backward_buffer = DiscriminationBuffer(buffer)

    forward_disc_losses = []
    forward_trans_losses = []
    backward_disc_losses = []
    backward_trans_losses = []

    for epoch_n in tqdm(range(1, n_epochs+1)):
        print("epoch", epoch_n)
        torch.cuda.empty_cache()

        forward_disc_epoch_losses = []
        forward_trans_epoch_losses = []
        forward_real_epoch_scores = []
        forward_fake_epoch_scores = []
        forward_photo_epoch_scores = []
        backward_disc_epoch_losses = []
        backward_trans_epoch_losses = []
        backward_real_epoch_scores = []
        backward_fake_epoch_scores = []
        backward_icon_epoch_scores = []
        cycle_consistency_losses = []

        forward_buffer.put(forward_model["trans"].state_dict()) # no deepcopy here
        backward_buffer.put(forward_model["trans"].state_dict()) # no deepcopy here
        for forward_icon_batch, backward_photo_batch in tqdm(zip(forward_icon_loader, backward_photo_loader)):
            forward_icon_batch, backward_photo_batch = forward_icon_batch.to(device), backward_photo_batch.to(device)
            disc_batch_losses, trans_batch_losses, real_batch_scores, fake_batch_scores, photo_batch_scores = simple_train_batch(
                    forward_model, forward_buffer, forward_photo_loader, forward_icon_batch, forward_loss_fn, forward_opt,
                    forward_d_loss_boundary, forward_t_loss_boundary, max_iter, lbl_noise,
                    device=device,
                )
            forward_disc_epoch_losses += disc_batch_losses
            forward_trans_epoch_losses += trans_batch_losses
            forward_real_epoch_scores += real_batch_scores
            forward_fake_epoch_scores += fake_batch_scores
            forward_photo_epoch_scores += photo_batch_scores

            disc_batch_losses, trans_batch_losses, real_batch_scores, fake_batch_scores, icon_batch_scores = simple_train_batch(
                    backward_model, backward_buffer, backward_icon_loader, backward_photo_batch, backward_loss_fn, backward_opt,
                    backward_d_loss_boundary, backward_t_loss_boundary, max_iter, lbl_noise,
                    device=device,
                )
            backward_disc_epoch_losses += disc_batch_losses
            backward_trans_epoch_losses += trans_batch_losses
            backward_real_epoch_scores += real_batch_scores
            backward_fake_epoch_scores += fake_batch_scores
            backward_icon_epoch_scores += icon_batch_scores

            cycle_opt.zero_grad()
            iconed = forward_model["trans"](backward_photo_batch) # nevermind forward-backward mixing
            forward_doppler = backward_model["trans"](iconed)
            forward_cycle_loss = cycle_coef*(harmonic_loss(dist_fn, backward_photo_batch, iconed, sigma=sigma, device=device) + \
                    harmonic_loss(dist_fn, iconed, forward_doppler, sigma=sigma, device=device))
            photoed = backward_model["trans"](forward_icon_batch)
            backward_doppler = forward_model["trans"](photoed)
            backward_cycle_loss = cycle_coef*(harmonic_loss(dist_fn, forward_icon_batch, photoed, sigma=sigma, device=device) + \
                    harmonic_loss(dist_fn, photoed, backward_doppler, sigma=sigma, device=device))
            cycle_loss = forward_cycle_loss + backward_cycle_loss
            cycle_loss.backward()
            cycle_opt.step()
            cycle_consistency_losses.append(cycle_loss.item())

        forward_buffer.items[-1] = deepcopy(forward_model["trans"].state_dict())
        backward_buffer.items[-1] = deepcopy(backward_model["trans"].state_dict())

        forward_epochal_disc_loss = np.mean(forward_disc_epoch_losses)
        forward_disc_losses.append(forward_epochal_disc_loss)
        forward_epochal_trans_loss = np.mean(forward_trans_epoch_losses)
        forward_trans_losses.append(forward_epochal_trans_loss)
        print(f"forward disc loss: {forward_epochal_disc_loss:.4f}, forward trans loss: {forward_epochal_trans_loss:.4f}")
        forward_epochal_real_score = np.mean(forward_real_epoch_scores)
        forward_epochal_fake_score = np.mean(forward_fake_epoch_scores)
        forward_epochal_photo_score = np.mean(forward_photo_epoch_scores)
        print(f"forward real score: {forward_epochal_real_score:.4f}, forward fake score: {forward_epochal_fake_score:.4f}, forward photo score: {forward_epochal_photo_score:.4f}")
        backward_epochal_disc_loss = np.mean(backward_disc_epoch_losses)
        backward_disc_losses.append(backward_epochal_disc_loss)
        backward_epochal_trans_loss = np.mean(backward_trans_epoch_losses)
        backward_trans_losses.append(backward_epochal_trans_loss)
        print(f"backward disc loss: {backward_epochal_disc_loss:.4f}, backward trans loss: {backward_epochal_trans_loss:.4f}")
        backward_epochal_real_score = np.mean(backward_real_epoch_scores)
        backward_epochal_fake_score = np.mean(backward_fake_epoch_scores)
        backward_epochal_icon_score = np.mean(backward_icon_epoch_scores)
        print(f"backward real score: {backward_epochal_real_score:.4f}, backward fake score: {backward_epochal_fake_score:.4f}, backward icon score: {backward_epochal_icon_score:.4f}")
        print(f"harmonic loss: {np.mean(cycle_consistency_losses):.4f}")

        if lr_sched is not None:
            if "disc" in lr_sched:
                lr_sched["disc"].step()
            if "trans" in lr_sched:
                lr_sched["trans"].step()

        print("Forward test")
        simple_draw_interm(forward_model, forward_test_loader)
        plt.show()
        print("Backward test")
        simple_draw_interm(backward_model, backward_test_loader)
        plt.show()

