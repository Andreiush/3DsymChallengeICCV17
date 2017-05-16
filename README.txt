Local 3D symmetry scene creator, created in Ubuntu 16.04:

Dependencies: boost, Eigen3, c++11

Installation:

mkdir bin
cd bin
cmake ..
make

Execution:

./createScene runs with the following options:

  --help                           Produce help message
  --modelList arg (=train.txt)     Axis aligned model file list
  --numberScenes arg (=200)        Number of scenes to generate
  --inputFolder arg (=aligned)     Input folder where the axis aligned models 
                                   are located
  --outputFolder arg (=localScene) Output folder

The list of axis aligned models from the provided training set is included in this package under the name "train.txt".

Additionally, a table model and the text file with its symmetry planes is attached (table.ply and table-plane.txt). These files HAVE TO BE on the same folder from where the program createScene is executed.

Example, running the code from the installation folder:

/bin/createScene --numberScenes 10 --modelList train.txt --inputFolder ~/data/symChallenge/3D-globalSym-synth-training/axis-aligned-models --outputFolder ~/data/symChallenge/generatedLocalScenes

Will use the training models located in ~/data/symChallenge/3D-globalSym-synth-training/axis-aligned-models and create 10 3D scenes and its groundtruth planes (under the name i-scene.ply i-scene-plane.txt i-scene-plane.wrl) in the folder ~/data/symChallenge/generatedLocalScenes
