import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from action_detect.net import NetV2

def openpose_to_jit():

    x = torch.randn(1,3,256,456)

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(r'D:\py\openpose_lightweight\weights\checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    net.eval()
    net(x)
    script_model = torch.jit.trace(net, x)
    script_model.save('test.jit')

def test_openpose_jit():
    x = torch.randn(1, 3, 256, 456)
    model = torch.jit.load(r"D:\py\openpose_lightweight\scripts\test.jit")

    print(model(x))

def action_to_jit():
    x = torch.randn(1, 16384)

    net = NetV2()
    checkpoint = torch.load(r'D:\py\openpose_lightweight\action_detect\checkPoint\action.pt', map_location='cpu')
    net.load_state_dict(checkpoint)
    net.eval()
    net(x)
    script_model = torch.jit.trace(net, x)
    script_model.save('action.jit')

    print(x.shape)

def test_action_jit():
    x = torch.randn(1, 16384)
    model = torch.jit.load(r"D:\py\openpose_lightweight\scripts\action.jit")

    print(model(x))


if __name__ == '__main__':
    test_action_jit()