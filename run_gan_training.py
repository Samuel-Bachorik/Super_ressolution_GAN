from process_dataset_2 import Process_dataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.optim import lr_scheduler
from graphing_class import CreateGraph
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
torch.backends.cudnn.benchmark = True
from torch import Tensor
import torchvision.models as models
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self) -> None:
        super(ContentLoss, self).__init__()

        vgg19 = models.vgg19(pretrained=True).eval()

        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])

        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # Normalization
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss


def main():
    batch_size          = 38
    best_loss_train     = 99999.0
    best_psnr_train     = 0.0
    best_loss_val       = 99999.0
    best_psnr_val       = 0.0
    epochs              = 9
    upsample            = 4
    training_path       = ("C:/Users/Samuel/PycharmProjects/Super_ressolution/cropped")
    validate_path       = ("C:/Users/Samuel/PycharmProjects/Super_ressolution/validate_crop")

    loader = Process_dataset(in_ress=64, out_ress=64 * upsample, training_path=training_path,
                             aug_count=2, validate_path=validate_path)

    batch_count_train   = (loader.get_training_count() + batch_size) // batch_size
    batch_count_val     = (loader.get_validate_count() + batch_size) // batch_size
    loss_chart_gen      = CreateGraph(batch_count_train, "Generator loss")
    loss_chart_disc     = CreateGraph(batch_count_train, "Discriminator loss")
    psnr_chart_val      = CreateGraph(batch_count_val, "Validate PSNR")

    generator           = Generator().to("cuda")
    PATH                = './Ich_generator_PSNR.pth'
    generator.load_state_dict(torch.load(PATH))

    discriminator       = Discriminator().to("cuda")

    g_optimizer         = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    d_optimizer         = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

    d_scheduler         = lr_scheduler.StepLR(d_optimizer, epochs // 2, 0.1)
    g_scheduler         = lr_scheduler.StepLR(g_optimizer, epochs // 2, 0.1)

    psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion = define_loss()

    scaler = amp.GradScaler()


    for epoch in range(epochs):
        psnr_train, loss_train = train_model(generator, discriminator, g_optimizer, d_optimizer,pixel_criterion,
                                             content_criterion, adversarial_criterion,loader, batch_size, scaler
                                             ,best_psnr_train, best_loss_train, loss_chart_gen, loss_chart_disc,
                                             batch_count_train, epoch)
        if psnr_train > best_psnr_train:
            best_psnr_train = psnr_train
        if loss_train < best_loss_train:
            best_loss_train = loss_train

        avg_psnr_val, avg_loss_val = validate_model(generator, pixel_criterion, content_criterion, loader, batch_size,
                                            batch_count_val,psnr_chart_val, epoch)
        d_scheduler.step()
        g_scheduler.step()
        loss_chart_gen.count(epoch)
        loss_chart_disc.count(epoch)
        psnr_chart_val.count(epoch)

        if avg_psnr_val > best_psnr_val:
            save_model(generator,'./GAN_generator_validate_PSNR_GIT.pth')
            best_psnr_val = avg_psnr_val

        if avg_loss_val < best_loss_val:
            save_model(generator, './GAN_generator_validate_loss_GIT.pth')
            best_loss_val = avg_loss_val

        save_model(generator,'./Generator_SR_epoch_{}_GIT.pth'.format(epoch + 1))


def save_model(model,path,):
    torch.save(model.state_dict(), path)
    print("Model {} saved ".format(path))

def define_loss() -> [nn.MSELoss, nn.MSELoss, ContentLoss, nn.BCEWithLogitsLoss]:
    psnr_criterion          = nn.MSELoss().to("cuda")
    pixel_criterion         = nn.MSELoss().to("cuda")
    content_criterion       = ContentLoss().to("cuda")
    adversarial_criterion   = nn.BCEWithLogitsLoss().to("cuda")

    return psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion


def train_model(generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                pixel_loss,
                content_loss,
                adversarial_loss,
                loader,
                batch_size,
                scaler,
                best_psnr_train,
                best_loss_train,
                loss_chart_gen,
                loss_chart_disc,
                batch_count,
                epoch):
    discriminator.train()
    generator.train()
    best_loss = best_loss_train
    best_psnr = best_psnr_train
    for batch in range(batch_count):
        lr, hr = loader.get_training_batch(batch_size)

        hr = hr.to("cuda")
        lr = lr.to("cuda")

        real_label = torch.ones((lr.size(0), 1)).to("cuda")
        fake_label = torch.zeros((lr.size(0), 1)).to("cuda")

        sr = generator(lr)

        PSNR = 10. * torch.log10(1. / (((sr - hr) ** 2).mean()))

        if PSNR.item() > best_psnr:
            best_psnr = PSNR.item()
            save_model(generator, './GAN_generator_train_PSNR_GIT.pth')

        for p in discriminator.parameters():
            p.requires_grad = True

        # Initialize the discriminator optimizer gradient
        d_optimizer.zero_grad()

        # Calculate the loss of the discriminator on the high-resolution image
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_loss(hr_output, real_label)
        # Gradient zoom

        scaler.scale(d_loss_hr).backward()

        # Calculate the loss of the discriminator on the super-resolution image.
        with amp.autocast():
            sr_output = discriminator(sr.detach())
            d_loss_sr = adversarial_loss(sr_output, fake_label)

        # Gradient zoom
        scaler.scale(d_loss_sr).backward()
        # Update discriminator parameters
        scaler.step(d_optimizer)
        scaler.update()

        # Count discriminator total loss
        d_loss = d_loss_hr + d_loss_sr

        loss_chart_disc.num_for_avg += float(d_loss.item())

        for p in discriminator.parameters():
            p.requires_grad = False

        g_optimizer.zero_grad()

        with amp.autocast():
            output = discriminator(sr)

            pixel_loss_tensor = 1.0 * pixel_loss(sr, hr.detach())
            content_loss_tensor = 1.0 * content_loss(sr, hr.detach())
            adversarial_loss_tensor = 0.001 * adversarial_loss(output, real_label)

        # Count discriminator total loss
        g_loss = pixel_loss_tensor + content_loss_tensor + adversarial_loss_tensor
        # Gradient zoom

        if g_loss.item() < best_loss:
            best_loss = g_loss.item()
            save_model(generator, './GAN_generator_train_loss_GIT.pth')

        scaler.scale(g_loss).backward()
        # Update generator parameters
        scaler.step(g_optimizer)
        scaler.update()

        loss_chart_gen.num_for_avg += float(g_loss.item())

    return best_psnr,best_loss


def validate_model(generator,
                   pixel_loss,
                   content_loss,
                   loader,
                   batch_size,
                   batch_count,
                   psnr_chart_val,
                   epoch):
    with torch.no_grad():
        generator.eval()
        avg_psnr = 0
        avg_loss = 0
        for batch in range(batch_count):
            lr, hr = loader.get_validate_batch(batch_size)
            lr, hr = lr.to("cuda"), hr.to("cuda")

            hr_pred = generator(lr)

            PSNR = 10. * torch.log10(1. / (((hr_pred - hr) ** 2).mean()))

            pixel_loss_tensor = 1.0 * pixel_loss(hr_pred, hr.detach())
            content_loss_tensor = 1.0 * content_loss(hr_pred, hr.detach())

            g_loss = pixel_loss_tensor + content_loss_tensor

            avg_psnr+= PSNR.item()
            avg_loss+= g_loss.item()
            psnr_chart_val.num_for_avg += PSNR.item()

    return avg_psnr/batch_count, avg_loss/batch_count #Aritmetic mean


if __name__ == "__main__":
    main()
