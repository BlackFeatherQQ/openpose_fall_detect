import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from action_detect.net import NetV2

def openpose_to_jit():

    x = torch.randn(1,3,256,456)

    net = PoseEstimationWithMobileNet().cpu()
    checkpoint = torch.load(r'.\weights\checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    net.eval()
    net(x)
    script_model = torch.jit.trace(net, x)
    script_model.save('test.jit')

def test_openpose_jit():
    x = torch.randn(1, 3, 256, 456)
    model = torch.jit.load("test.jit")

    print(model(x))

def action_to_jit():
    x = torch.randn(1, 16384)

    net = NetV2().cpu()
    checkpoint = torch.load(r'.\action_detect\checkPoint\action.pt', map_location='cpu')
    net.load_state_dict(checkpoint)
    net.eval()
    net(x)
    script_model = torch.jit.trace(net, x)
    script_model.save('action.jit')

    print(x.shape)

def test_action_jit():
    x = torch.randn(1, 16384)
    model = torch.jit.load("action.jit")

    print(model(x))


if __name__ == '__main__':
    test_action_jit()
