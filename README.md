# Stacked hourglass with CoordConvGRU RNN cells
![CellGIF](https://i.imgur.com/5w37uXz.gif) 

## Usage

### Generating TFRecords

#### Annotations file format:  
```
seq1_frame1.png x1 y1 x2 y2 ... xM yM seq1_frame2.png x1 y1 x2 y2 ... xM yM seq1_frameN.png x1 y1 x2 y2 ... xM yM
seq2_frame1.png x1 y1 x2 y2 ... xM yM seq2_frame2.png x1 y1 x2 y2 ... xM yM seq2_frameN.png x1 y1 x2 y2 ... xM yM
.
.
.
```
Execute the command below to to generate **train.tfrecords** and **train.tfrecords** files

`python generate_tfrecords.py --data-dir='dataset_folder' --input-file `annotations_file.txt` --seq-length 4`

### Training

Use the following command to start the training session:

`python hg_main.py --data-dir='tfrecord_dir' --job-dir='checkpoint_dir' --num-gpus=4 --train-batch-size=16 --eval-batch-size=16 --seq-length=4 --train-steps=500000 --variable-strategy GPU`

It will use **train.tfrecords** and **eval.tfrecords** files in the tfrecord_dir path to training and validation respectively

### Testing
`python hg_main.py --data-dir='tfrecord_dir' --job-dir='checkpoint_dir' --mode='test' --eval-steps=1000`

### Project Page
[https://github.com/ayushgaud/Stacked_HG_CoordConvGRU](https://github.com/ayushgaud/Stacked_HG_CoordConvGRU/blob/master/index.md)
