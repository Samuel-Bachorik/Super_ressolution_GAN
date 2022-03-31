from process_dataset_2 import Process_dataset
from generator_model import Generator
from graphing_class import CreateGraph
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
torch.backends.cudnn.benchmark = True

def main():
    batch_size      = 32
    best_loss_train = 99999.0
    best_psnr_train = 0.0
    best_loss_val   = 99999.0
    best_psnr_val   = 0.0
    epochs          = 9
    upsample        = 4
    training_path   = ("C:/Users/Samuel/PycharmProjects/Super_ressolution/dataset4")
    validate_path   = ("C:/Users/Samuel/PycharmProjects/Super_ressolution/datik")

    loader = Process_dataset(in_ress=64, out_ress=64 * upsample, training_path=training_path,
                             aug_count=2, validate_path=validate_path)

    batch_count_train   = (loader.get_training_count() + batch_size) // batch_size
    batch_count_val     = (loader.get_validate_count() + batch_size) // batch_size

    loss_chart_gen      = CreateGraph(batch_count_train, "Train MSE loss")
    psnr_chart_train    = CreateGraph(batch_count_train, "Tain PSNR")

    loss_chart_val      = CreateGraph(batch_count_val, "Validate MSE loss")
    psnr_chart_val      = CreateGraph(batch_count_val, "Validate PSNR")

    mse_loss            = nn.MSELoss()
    generator = Generator().to("cuda")
    PATH      = './Ich_generator_PSNR.pth'
    generator.load_state_dict(torch.load(PATH))

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))

    scaler = amp.GradScaler()

    for epoch in range(epochs):
        psnr_train, loss_train = train_model(generator, g_optimizer,mse_loss, loader, batch_size, scaler,
                                             best_loss_train,best_psnr_train, batch_count_train,
                                             psnr_chart_train,loss_chart_gen)

        if psnr_train > best_psnr_train:
            best_psnr_train = psnr_train

        if loss_train < best_loss_train:
            best_loss_train = loss_train

        print(best_psnr_train,"PSNR train")
        print(best_loss_train,"Loss train")
        print("\n")

        avg_psnr_val, avg_loss_val = validate_model(generator, mse_loss,loader, batch_size,
                                                    batch_count_val,psnr_chart_val,loss_chart_val)

        loss_chart_gen.count(epoch)
        psnr_chart_train.count(epoch)
        loss_chart_val.count(epoch)
        psnr_chart_val.count(epoch)

        if avg_psnr_val > best_psnr_val:
            #save_model(generator,'./GAN_generator_validate_PSNR_GIT.pth')
            best_psnr_val = avg_psnr_val

        if avg_loss_val < best_loss_val:
            #save_model(generator, './GAN_generator_validate_loss_GIT.pth')
            best_loss_val = avg_loss_val
        print(best_psnr_val, "PSNR val")
        print(best_loss_val, "Loss val")

        #save_model(generator,'./Generator_SR_epoch_{}_GIT.pth'.format(epoch + 1))


def save_model(model,path,):
    torch.save(model.state_dict(), path)
    print("Model {} saved ".format(path))



def train_model(generator,
                g_optimizer,
                mse_loss,
                loader,
                batch_size,
                scaler,
                best_loss_train,
                best_psnr_train,
                batch_count,
                psnr_chart_train,
                loss_chart_gen):


    generator.train()
    best_loss = best_loss_train
    best_psnr = best_psnr_train

    for batch in range(batch_count):
        lr, hr = loader.get_training_batch(batch_size)

        hr = hr.to("cuda")
        lr = lr.to("cuda")

        g_optimizer.zero_grad()
        with amp.autocast():
            sr = generator(lr)

            loss = mse_loss(sr,hr)

            PSNR = 10. * torch.log10(1. / loss)

        loss_number = loss.item()
        psnr_number = PSNR.item()

        loss_chart_gen.num_for_avg    += loss_number
        psnr_chart_train.num_for_avg  += psnr_number

        scaler.scale(loss).backward()
        # Update generator parameters
        scaler.step(g_optimizer)
        scaler.update()

        if psnr_number > best_psnr:
            #save_model(generator,'./MSE_generator_train_PSNR_GIT.pth')
            best_psnr = psnr_number

        if loss_number < best_loss:
            #save_model(generator, './MSE_generator_train_loss_GIT.pth')
            best_loss = loss_number

    return best_psnr, best_loss



def validate_model(generator,
                   mse_loss,
                   loader,
                   batch_size,
                   batch_count,
                   psnr_chart_val,
                   loss_chart_val):

    with torch.no_grad():
        generator.eval()
        avg_psnr = 0
        avg_loss = 0
        for batch in range(batch_count):
            lr, hr = loader.get_validate_batch(batch_size)
            lr, hr = lr.to("cuda"), hr.to("cuda")

            hr_pred = generator(lr)

            PSNR = 10. * torch.log10(1. / (((hr_pred - hr) ** 2).mean()))
            loss = mse_loss(hr_pred,hr)

            avg_psnr+=PSNR.item()
            avg_loss+=loss.item()

            psnr_chart_val.num_for_avg+=PSNR.item()
            loss_chart_val.num_for_avg+=loss.item()

    return avg_psnr/batch_count, avg_loss/batch_count #Aritmetic mean


if __name__ == "__main__":
    main()