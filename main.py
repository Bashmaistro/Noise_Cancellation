import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Audio path
path = 'Sounds/*'

sound_files = glob.glob(path)

output = 'spektrograms'

if not os.path.exists(output):
    os.makedirs(output)

for files in sound_files:
    # Lead auido file
    audio, sampling_rate = librosa.load(files)

    # Create Spektrogram
    spektrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate,
                                                 hop_length=32, n_fft=512)

    # Turn into a image
    plt.figure(figsize=(16, 8), dpi=100)
    librosa.display.specshow(librosa.power_to_db(spektrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.axis('off')

    file_name = os.path.basename(files).replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')
    save_path = os.path.join(output, f'{file_name}_spektrogram.png')

    # Save spektogram
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f'{file_name} spektrogram is saved.')


