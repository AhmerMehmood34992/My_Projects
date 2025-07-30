import librosa
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
aud,sr,labels=[],[],[]

import librosa
import numpy as np
import os
# path = r"B:\MY AI\Audio_preprocessing_funcs\data"

# for class_folder in os.listdir(path):
#     class_path = os.path.join(path, class_folder)

#     if os.path.isdir(class_path):  # Ensure it's a folder
#         # Loop over files inside class folder
#         for file in os.listdir(class_path):
#             if file.endswith(".wav") or file.endswith(".mp3"):
#                 file_path = os.path.join(class_path, file)
#                 print(f"Loading {file_path}")
                
#                 a, s = librosa.load(file_path, sr=None)  # Keep original sampling rate
#                 aud.append(a)
#                 sr.append(s)
#                 labels.append(class_folder)  # Use folder name as label
                




# processed_audios = []
# for file in aud:
#     audio_trimmed, _ = librosa.effects.trim(file, top_db=20)
#     processed_audios.append(audio_trimmed)
   
# print("trim dome")
# # np.savez_compressed("audio_dataset_proces.npz", aud=processed_audios, sr=sr, labels=labels)
     
# aud_normalized=[]      
# for aud in processed_audios:       
#     norm_audio = librosa.util.normalize(aud, norm=np.inf)
#     aud_normalized.append(norm_audio)

# # np.savez_compressed("audio_dataset_norm.npz", aud=aud_normalized, sr=sr,labels=labels)
# print("norm dome")
                 





# # max_amplitude = np.max(np.abs(audio))
# #     if max_amplitude > 0:  # Avoid division by zero
# #         return audio / max_amplitude
# #     return audio



# #  Find max length

# # Pad or truncate to target duration using librosa.util.fix_length
# #  target_samples = int(target_duration * sr)
# #  aud_fixed = librosa.util.fix_length(aud_normalized, size=target_samples)


# max_len = max(len(x) for x in aud_normalized)

# # Pad all to max_len
# aud_padded = [np.pad(x, (0, max_len - len(x))) for x in aud_normalized]
# print("pad dome")

# # Convert to 2D NumPy array
# # aud_padded = np.stack(aud_padded).astype(np.float16)
# import numpy as np

# # Ensure arrays are properly formatted
# aud_padded = np.array(aud_padded, dtype=np.float16)  # Audio data
# labels = np.array(labels, dtype=np.int16)  # Labels
# sr = np.array([sr], dtype=np.int32)  # Sample rate as a single-element array

# # Save to compressed file
# np.savez_compressed("audio_dataset_padded.npz", aud=aud_padded, sr=sr, labels=labels)
# # np.savez_compressed("audio_dataset_padded.npz", aud=aud_padded, sr=sr, labels=labels)

# print("Saved dataset successfully!")


data = np.load("audio_dataset_padded.npz")
aud_processed = data['aud']  # Shape: (n_files, target_samples)
sr = data['sr'][0] # Assuming all files have the same sample rate
labels = data['labels']
def extract_mfcc(audio, sr, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr[0], n_mfcc=n_mfcc)
    return mfccs

mfccs_list = [extract_mfcc(audio, sr) for audio in aud_processed]


# # import librosa
# # import numpy as np

mfcc_list = []
for x in aud_processed:  # each x is a 1D audio signal
    mfcc = librosa.feature.mfcc(y=x, sr=sr[0], n_mfcc=40)
    mfcc_list.append(mfcc)

# # print("mfcc done")

max_len = max([mf.shape[1] for mf in mfcc_list])
# mfcc_padded = [np.pad(mf, ((0, 0), (0, max_len - mf.shape[1])), mode='constant') for mf in mfcc_list]
# mfcc_array = np.array(mfcc_padded)
# # print(mfcc_padded.shape)  # (n_files, n_mfcc, target_samples
# np.savez_compressed("mfcc_dataset.npz", mfccs=mfcc_array, labels=labels)
# print("MFCC extraction and padding completed successfully!")


# data = np.load("mfcc_dataset.npz")
# mfccs = data["mfccs"]
# labels = data["labels"]

# # # # Reshape to 2D: (n_samples, n_mfcc * n_frames)
# n_samples, n_mfcc, n_frames = mfccs.shape
# mfccs_reshaped = mfccs.reshape(n_samples, -1)

# scaler = StandardScaler()
# mfccs_scaled = scaler.fit_transform(mfccs_reshaped)
# mfccs_scaled = mfccs_scaled.reshape(n_samples, 40, 93, 1)

# np.savez_compressed("data_compressesd.npz",mfcc=mfccs_scaled,labels=labels)
# print("saved")


# data = np.load("data_compressesd.npz")
# mfccs = data["mfcc"]
# labels = data["labels"]
# # print(data.files)


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=42)


# y_train_enc = to_categorical(y_train, num_classes=10)
# y_test_enc = to_categorical(y_test, num_classes=10)


# print(X_test[:5])

# model=Sequential()
# # print(X_train[0].shape)
# model.add(Conv2D(32,(3,3),activation='relu',input_shape=X_train[0].shape))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))


# model.add(Flatten())
# model.add(Dense(64,activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(10,activation='softmax'))
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# history=model.fit(X_train,y_train_enc,epochs=10,batch_size=32,validation_data=(X_test,y_test_enc))

# res=model.predict(X_test)
# y_pred=np.argmax(res,axis=1)

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])

# plt.plot(history.history["accuracy"])
# plt.plot(history.history["val_accuracy"])

# plt.show()

# model.save("mnist_aud_model.h5")
import tensorflow as tf
model=tf.keras.models.load_model("B:\My AI\mnist_aud_model.h5")
import librosa
audio_path = r"B:\My AI\Audio_preprocessing_funcs\one.wav"
audio, sr = librosa.load(audio_path, sr=None)  # Use same sample rate as training

# Normalize
audio = audio / max(abs(audio))

# Optional: pad to max length used during training
# max_len = 8000  # use same as training
if len(audio) < max_len:
    audio = np.pad(audio, (0, max_len - len(audio)))
else:
    audio = audio[:max_len]

# Extract MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

# Ensure consistent shape (40, time_steps)
import numpy as np
mfcc = mfcc[:, :93]  # or pad/truncate to fixed time steps (93 is common if that’s what your model expects)

# Reshape and expand dims to match input shape (batch_size, 40, 93, 1)
mfcc = mfcc.reshape((40, 93, 1))
mfcc = np.expand_dims(mfcc, axis=0)  # (1, 40, 93, 1)

prediction = model.predict(mfcc)
predicted_class = prediction.argmax(axis=1)[0]
print(f"Predicted digit: {predicted_class}")
