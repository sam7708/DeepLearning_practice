"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import json
from src.deep_q_network import DeepQNetwork
from src.ns_shaft import NS_SHAFT
from src.utils import pre_processing
import matplotlib.pyplot as plt
import numpy as np
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play NS-SHAFT""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt,testname, result):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load(('{}/' + testname).format(opt.saved_path))
    else:
        model = torch.load("{}/ns_shaft".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    game_state = NS_SHAFT()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(400)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    total_score = 0
    while terminal==False:
        prediction = model(state)
        action = (prediction.data.max(1)[1].item()-1)*4

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(400)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        total_score +=1
        state = next_state
    result.append(total_score)
    print(total_score)

if __name__ == "__main__":
    opt = get_args()
    iters = 50
    results = {}
    result = []
    for i in range(iters):
        test(opt,'ns_shaft_100',result)
    results['ns_shaft_100'] = result

    result = []
    for i in range(iters):
        test(opt,'ns_shaft_100000',result)
    results['ns_shaft_100000'] = result

    result = []
    for i in range(iters):
        test(opt,'ns_shaft_120000',result)
    results['ns_shaft_120000'] = result

    result = []
    for i in range(iters):
        test(opt,'ns_shaft_500000',result)
    results['ns_shaft_500000'] = result

    x = np.arange(0,iters)
    for i in results:
        y = results[i]
        print(y,i)
        plt.plot(x,y, label=i) 
    plt.xlabel("Iterations") 
    plt.ylabel("Rewards")
    plt.legend()
    plt.show()
