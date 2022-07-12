import subprocess
import sys
import os
from tqdm import tqdm

# script for consecutively training all the methods with different hyperparameter sets detailed in train_mnist and train_sine

mnist = False
sine = True

path = os.path.dirname(os.path.realpath(__file__))
parnums = [0, 3, 4, 8, 12]

if mnist:
    ### train mnist ###
    try:
        cmd = subprocess.run([sys.executable, path+"/train_mnist.py", '-1'], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(path+"/train_mnist.py")

    stdout = cmd.stdout.decode()
    num_pars = int(stdout)
    mnist_errors = [0]*num_pars

    for i in tqdm(parnums):
        try:
            # print("Running mnist with parameters " + str(i) + ":")
            cmd = subprocess.run([sys.executable, path+"/train_mnist.py", str(i)], capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            print(cmd.stdout.decode())
            print(cmd.stderr.decode())
            mnist_errors[i] = 1

if sine:
    ### train sine ###

    try:
        cmd = subprocess.run([sys.executable, path+"/train_sine.py", '-1'], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(print(path+"/train_sine.py"))

    stdout = cmd.stdout.decode()
    num_pars = int(stdout)
    sine_errors = [0]*num_pars

    for i in tqdm(parnums):
        try:
            # print("Running sine with parameters " + str(i) + ":")
            cmd = subprocess.run([sys.executable, path+"/train_sine.py", str(i)], capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            print(cmd.stdout.decode())
            print(cmd.stderr.decode())
            sine_errors[i] = 1

print("")
print("")

if mnist:
    if sum(mnist_errors) > 0:
        print("There were errors during the mnist training with:")
        for i in range(len(mnist_errors)):
            if mnist_errors[i]==1:
                print("   parameter set " + str(i))

if sine:
    if sum(sine_errors) > 0:
        print("There were errors during the sine training with:")
        for i in range(len(sine_errors)):
            if sine_errors[i]==1:
                print("   parameter set " + str(i))
