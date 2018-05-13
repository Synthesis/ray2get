# Ray2Get 2.0

Initially written in C and released back in 2009, Ray2Get is a tool that converts .apm audio files from the PC version of Rayman 2 to .wav files.

This new version, written in Python for convenience (cross-platform environment, no recompiling required after every tweak), fixes inaccuracies in the decoding process, supports more formats (see *Features* section) and now also allows to convert from .wav to .apm!

This means that you can replace the game's soundtrack with your own music! But to do so, you will need to split your files in parts and rename them accordingly. The file named `file_sequences.txt` lists all the sequences played in the game at every location (without the `.apm` extensions).


### Inaccurate ADPCM decoding?

The DLL provided with the Rayman 2 executable (`Rayman2\DLL\APMmxBVR.dll`) doesn't implement the IMA-ADPCM algorithm properly: each time a sample is processed, the ADPCM *step* variable has its least significant 3 bits cleared for some reason. This results in DC offset progressively building up at times when processing files encoded with the regular IMA-ADPCM algorithm (and so are most of the .apm files provided with the game).

As a workaround, this script provides a variant of the encoding algorithm that cancels out the side effect (by clearing the lowest 3 bits of the *step* in the encoding process as well). The tool uses the "inaccurate" version by default, for both encoding and decoding. Add the `i` parameter to use the standard IMA ADPCM algorithm instead (see *Usage*).


## Features

* Two algorithms: standard IMA-ADPCM, a Rayman 2-specific variant (see previous section);

* Now supports any sample rate (although the game will resample to 22050 Hz);

* Supports mono and stereo files;

* Supports 8-bit, 16-bit, 24-bit and 32-bit integer WAV as input files (floating point is not supported). The output files are always 16-bit;

* When decoding, the script discards the first PCM sample (stored in the .apm header) by default (as the game itself does). Use the `d1` option to keep it in the output file.


## Requirements

The script requires a Python 2.x environment installed on your machine (it was only tested in Python 2.7, and might not work with previous versions).


## Usage

`python ray2get.py [options] input_file [output_file]`

The accepted options are the following:

* `e`: encode a .wav file to .apm using the in-game algorithm variant;
* `ei`: encode a .wav file to .apm using the standard IMA-ADPCM algorithm;
* `d`: decode an .apm file to .wav using Rayman 2's algorithm (default);
* `di`: decode an .apm file to .wav using the standard IMA-ADPCM algorithm;
* `d1`: decode an .apm file to .wav and keep the first PCM value in the output file;

For decoding, both `i` and `1` parameters are optional and interchangeable (for example, you may use either `di1` or `d1i` if you want to decode as IMA-ADPCM and keep the first sample).

The last parameter (output file name) is optional. If not specified, the output file is given the name of the input file with the extension replaced.


### Batch-process all audio files in a directory

*CMD (Windows)* .bat file example:

```
@echo off
echo Starting conversion...
for %%f in (*.[EXT]) do echo "%%f" & python ray2get.py [options] "%%f"
echo Converted.
@echo on
pause
```

*Bash* .sh script example:

```
#!/bin/bash

echo Starting conversion...
for f in *.[EXT]
do
	echo "${f}"
	python ray2get.py [options] "${f}"
done
echo Converted.
```

In either script, replace `[EXT]` with the input file extension (`wav` for encoding, `apm` for decoding), and `[options]` with whatever you need to do (`e` for encoding, `d` for decoding, etc., see previous section). Both `ray2get.py` and your shell script have to be in the same directory as the files you're trying to process. *(And don't forget to check cautionary note \#2 below!)*


### /!\\ Cautionary notes

1. The game engine doesn't support mono files as background music. If you're looking to replace the musics with your own, make sure your .wav files are stereo before converting.

2. Ray2Get won't ask for permission to overwrite any existing file. As an example, if you're trying to decode `foo.apm` without specifying an output file name, make sure you don't already have a file called `foo.wav` or else it will be overwritten!


## Special thanks to

The folks from the RaymanPC forum;

deton24 for pointing out that the game audio won't run at more than 22050 Hz, and (unknowingly?) helping me realize that it isn't accurate IMA ADPCM;

imaginaryPineapple (from the [OpenRayman](https://github.com/imaginaryPineapple/OpenRayman) project) for the complementary info about the APM file format.


**Resources on the ADPCM file format:**

* Article on MultimediaWiki: https://wiki.multimedia.cx/index.php/IMA_ADPCM

* Original IMA-ADPCM specification document (an OCR version in PDF is also available there): http://www.cs.columbia.edu/~hgs/audio/dvi/
