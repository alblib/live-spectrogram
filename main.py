import sounddevice as sd
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import signal

previous_buffer = np.zeros(1024)
fft_map = np.full((256, 256), -60, dtype=float)
arr = np.array(range(256))
xi = (-(arr + 1) * 1024 / 48000)[::-1]
yi = (arr / 256 * 6000)
X,Y = np.meshgrid(xi, yi)

fig, ax = plt.subplots(2)

# fig = plt.figure()

# Update function for the spectrogram plot
def update_spectrogram(indata, frames, time, status):
    global previous_buffer, fft_map, xi, yi, arr, X, Y, ax
    buffer = previous_buffer
    previous_buffer = np.array(indata[:, 0])
    buffer = np.concatenate((buffer, previous_buffer))

    # Blackman-Harris window
    n = np.array(range(2048)) * 2 * math.pi / 2048
    window = 0.35875 - 0.48829 * np.cos(n) + 0.14128 * np.cos(2 * n) - 0.01168 * np.cos(3 * n)
    windowed_buffer = np.multiply(buffer, window)
    # windowed_buffer = buffer  # if rectangular window, pitch cannot be determined. (f, 2f, 3f, ...)
    preemphasized_buffer = windowed_buffer - 0.9 * np.concatenate(([0], windowed_buffer[1:]))
    correlation = signal.correlate(preemphasized_buffer, preemphasized_buffer)[:2048][::-1]
    ax[1].clear()
    # ax[1].plot(range(int(48000/300), int(48000/60)), correlation[int(48000/300):int(48000/60)])
    ax[1].plot(range(0, int(48000/60)), correlation[0:int(48000/60)])
    # pitch_detection_range = (48000/250, 48000/80)
    max_index = max(range(int(48000/300), int(48000/60)), key=correlation.__getitem__) # 80,250->60,300
    # if correlation[int(48000/300)] * 1.5 < correlation[max_index]:
    if correlation[0] * 0.25 <= correlation[max_index]:
        print(48000 / max_index, correlation[max_index] / correlation[0])


    fft_line = 20 * np.log10(np.abs(np.fft.fft(preemphasized_buffer)))[:256]
    fft_map = np.concatenate((fft_map[1:], [fft_line]))

    # plt.clear()
    # plt.colorbar()

def animate(args):
    global fft_map, X, Y, ax
    ax[0].pcolormesh(X, Y, np.transpose(fft_map), shading='auto')

stream = sd.InputStream(callback=update_spectrogram,
                        channels=1,
                        samplerate=48000,
                        blocksize=1024)
stream.start()
anim = animation.FuncAnimation(fig, animate, frames=1000, interval=10)
plt.show()
sd.sleep(int(10 * 1000))
stream.stop()
stream.close()
