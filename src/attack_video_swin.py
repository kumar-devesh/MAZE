from models.VideoSwin import PretrainedVideoSwinTransformer
from models.ResNet3D import ResNet3D
from models.Generator import Generator
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import itertools

cudnn.enabled = True
cudnn.benchmark = True
device = torch.device('cuda')


########## HYPERPARAMETERS ##########

batch_size = 4
student_iter = 5
generator_iter = 5
experience_iter = 10
num_zoge_directions = 2
budget = 10e6

budget_per_iter = batch_size * ((student_iter - 1) + (1 + num_zoge_directions) * generator_iter)
total_num_iters = budget // budget_per_iter

total_num_iters = int(total_num_iters)

lr_student = 0.001
lr_generator = 0.001

generator_input_dim = 120

smoothing_factor = 0.001


########## MODELS ##########

# preparing teacher
teacher = PretrainedVideoSwinTransformer('checkpoints/swin_base_patch244_window877_kinetics400_1k.pth').to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# preparing student
student = ResNet3D(pretrained=False).to(device)
student_opt = optim.SGD(student.parameters(), lr=lr_student, momentum=0.9, weight_decay=5e-4)
student.train()

# preparing generator
generator = Generator()
generator_opt = optim.SGD(generator.parameters(), lr=lr_generator, momentum=0.9, weight_decay=5e-4)
generator.train()


########## Zeroth Order Gradient Estimation ##########

def backward_using_zoge(X, X_pre, student, teacher):
    gradient_estimate = torch.zeros_like(X_pre)
    dimensions = np.array(X.shape[1:]).prod()

    # stop student grad updates
    for p in student.parameters():
        p.requires_grad = False

    # estimate grad
    with torch.no_grad():
        student_preds = student(X)
        teacher_preds = teacher(X)

        # original kl divergence between student and teacher output distributions
        original_loss = -F.kl_div(
            F.log_softmax(student_preds, dim=1),
            F.softmax(teacher_preds, dim=1),
            reduction="none",
        ).sum(dim=1)

        for direction in range(num_zoge_directions):
            # random vector u for slight perturbations
            u = torch.randn(X_pre.shape, device=device)
            u_flattened = u.view([batch_size, -1])
            u_norm = u / torch.norm(u_flattened, dim=1).view([-1, 1, 1, 1, 1])

            # perturbed input to the student and teacher
            perturbed_X_pre = x_pre + (smoothing_factor * u_norm)
            perturbed_X = nn.Tanh()(perturbed_X_pre)

            # perturbed output logits
            student_preds = student(perturbed_X)
            teacher_preds = teacher(perturbed_X)

            # perturbed loss
            perturbed_loss = -F.kl_div(
                F.log_softmax(student_preds, dim=1),
                F.softmax(teacher_preds, dim=1),
                reduction="none",
            ).sum(dim=1)

            # gradient estimate from original and modified loss
            gradient_estimate += ((dimensions/num_zoge_directions) * (perturbed_loss-original_loss)/smoothing_factor).view([-1, 1, 1, 1, 1]) * u_norm
        
        # backward to update parameters using zoge
        gradient_estimate = gradient_estimate / batch_size
        X_pre.backward(gradient_estimate, retain_graph=True)

        for p in student.parameters():
            p.requires_grad = True


########## TRAIN LOOP ##########

stored_teacher_output_data = []
total_queried = 0

for i in range(total_num_iters):

    ########## GENERATOR ##########

    for g in range(generator_iter):

        # random embedding
        random_emb = torch.randn(batch_size, generator_input_dim).to(device)

        # generator forward pass
        x, x_pre = generator(random_emb)

        # generator backward pass using grad estimation
        generator_opt.zero_grad()
        backward_using_zoge(x, x_pre, student, teacher)
        generator_opt.step()

    ########## STUDENT ##########

    for c in range(student_iter):

        with torch.no_grad():
            # generate videos from generator
            z = torch.randn((batch_size, generator_input_dim), device=device)
            x, _ = generator(z)
            x = x.detach()

            # get teacher outputs for the generated videos
            teacher_preds = teacher(x)
        
        # stduent forward
        student_preds = student(x)

        # student backward
        loss = F.kl_div(
            F.log_softmax(student_preds, dim=1),
            F.softmax(teacher_preds, dim=1),
            reduction='batchmean',
        )
        student_opt.zero_grad()
        loss.backward()
        student_opt.step()

        # print results
        print(f'student pred: {student_preds.argmax(-1).item()}, teacher pred: {teacher_preds.argmax(-1).item()}')

    student_loss = loss

    ########## EXPERIENCE REPLAY ##########

    # storing previous datapoints in an experience dataloader
    current_teacher_output_data = [(a, b) for a, b in zip(x.cpu().detach().numpy(), teacher_preds.cpu().detach().numpy())]
    stored_teacher_output_data += current_teacher_output_data
    experience_dataloader = torch.utils.data.DataLoader(stored_teacher_output_data, batch_size=batch_size, shuffle=True)
    experience_dataloader_iter = itertools.cycle(experience_dataloader)

    experience_loss = torch.tensor(0.0, device=device)

    for e in range(experience_iter):
        x, teacher_preds = next(experience_dataloader_iter)
        if x.size(0) < batch_size:
            break
        x, teacher_preds = x.to(device), teacher_preds.to(device)

        # student forward
        student_preds = student(x)

        # student backward
        loss = F.kl_div(
            F.log_softmax(student_preds, dim=1),
            F.softmax(teacher_preds, dim=1),
            reduction='batchmean',
        )
        student_opt.zero_grad()
        loss.backward()
        student_opt.step()

        experience_loss += loss

    experience_loss = experience_loss / experience_iter

    ########## LOGGING ##########

    total_queried += budget_per_iter
    print(f'queried videos till now: {total_queried}, student loss: {student_loss}')

torch.save(student, 'student.pth')
torch.save(generator, 'generator.pth')