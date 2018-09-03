#!/usr/bin/env python

"""
	Ray2Get 2.0
	by Palorifgrodbierzrt 2018
	[ver. 2018-09-04]


Ten years later (or maybe ten years too late?), I'm back!
Here is an updated revision of my music-ripping tool for the PC version of
Rayman 2 (now written in Python for convenience).

This version fixes a minor inaccuracy in the decoding of APM files, and is much
more versatile. You can now convert from .apm to .wav... and back!

More formats are supported: mono and stereo files (too bad mono files can't be
played in the game, though...), any sample rate (but the game will resample to
22050Hz anyway), and it can read 8, 16, 24 and 32-bit integer PCM wav files
(floating point is not supported).

The game uses a slight variation of the standard IMA-ADPCM algorithm. Every
time a nibble is processed, the ADPCM step variable has its least significant 3
bits cleared, for some reason.
While decoding, the game ditches the first PCM sample (located in the APM
header), so I included two decoding modes: one which processes .apm files like
the game does, and another one which keeps the first sample in the output file
(in case you're a purist like me :p).


Usage:
  Encoding, for proper play in game:  python ray2get.py e file.wav [file.apm]
  Encoding, standard IMA-ADPCM:       python ray2get.py ei file.wav [file.apm]

  Decoding, as the game would:        python ray2get.py d file.apm [file.wav]
  Decoding, standard IMA-ADPCM:       python ray2get.py di file.apm [file.wav]
  Decoding, keep 1st sample:          python ray2get.py d1 file.apm [file.wav]

Flags 'i' and '1' for APM decoding are optional and can be both used
interchangeably at the same time.
The last parameter (output file name) is optional. If not provided, the output
file name will be the same as the input file name, with the extension replaced
with .apm (or appended if the input file has no extension).


Special thanks to the folks from the Rayman Pirate Community, deton24 for
(unknowingly) pointing some of my bugs out, imaginaryPineapple (from the
OpenRayman project) for the complementary info about the APM file format.
"""


from __future__ import print_function
import os.path, re, struct, sys


# -----------------------------------------------------------------------------
# Common data
# -----------------------------------------------------------------------------

def toList(value, bytes=4, little_endian=True):
	"""
		Converts an integer to a list of bytes (useful for file binary output).

		value: an integer value
		bytes: the number of bytes in the output list
		little_endian: if True, the list will contain the least significant
		               byte first
	"""
	v = value
	l = []
	for b in range(bytes):
		l.append((v & 255))
		v = v >> 8
	if little_endian == False:
		l = l[::-1] # flip the list backwards
	return l


# The IMA index table
index_table = [-1, -1, -1, -1, 2, 4, 6, 8]

# The IMA step table
step_table = [
	7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
	19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
	50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
	130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
	337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
	876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
	2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
	5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
	15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
]



# -----------------------------------------------------------------------------
# Encoder section
# -----------------------------------------------------------------------------

def encode_init(sample1, sample2):
	"""
		Initialize the ADPCM encoding, using the first two PCM samples of the
		audio signal.

		See http://www.cs.columbia.edu/~hgs/audio/dvi/ for more information.
	"""
	predictor = sample1
	diff = sample2 - sample1

	if diff < 0:
		diff = -diff
	elif diff > 32767:
		diff = 32767
	stepindex = 0
	while step_table[stepindex] < diff:
		stepindex += 1

	return predictor, stepindex


def encode(sample, predictor, stepindex, ima_adpcm=False):
	"""
		Perform one step of the ADPCM encoding algorithm, with given inputs.

		sample: the next PCM sample to encode
		predictor: the current PCM predictor value
		stepindex: the current ADPCM step index
		ima_adpcm: if False, the lowest 3 bits of the step variable will be
		           cleared

		See http://www.cs.columbia.edu/~hgs/audio/dvi/ for more information.
	"""
	delta = sample - predictor
	nibble = 0
	if delta < 0:
		nibble = 8
		delta = -delta

	step = step_table[stepindex]
	diff = step >> 3
	if not ima_adpcm:
		step &= (~7)

	if delta >= step:
		nibble |= 4
		delta -= step
		diff += step
	step >>= 1
	if delta >= step:
		nibble |= 2
		delta -= step
		diff += step
	step >>= 1
	if delta >= step:
		nibble |= 1
		diff += step

	if nibble & 8:
		predictor -= diff
	else:
		predictor += diff
	if predictor < -32768:
		predictor = -32768
	elif predictor > 32767:
		predictor = 32767

	stepindex += index_table[nibble & 7]
	if stepindex < 0:
		stepindex = 0
	elif stepindex > 88:
		stepindex = 88

	return nibble, predictor, stepindex


def read_sample_16(file_handler, bytes_per_sample):
	"""
		Reads a PCM sample from a file, and converts it to 16 bits.

		file_handler: a file handler to .read() the bytes from
		bytes_per_sample: an integer between 1 and 4 (only 8, 16, 24 and 32-bit
		                  files are supported)
	"""
	data = file_handler.read(bytes_per_sample)
	if data == '':
		return 0

	if bytes_per_sample == 1:
		return (struct.unpack('<B', data[:1])[0] << 8)
	elif bytes_per_sample == 2:
		return struct.unpack('<h', data[:2])[0]
	elif bytes_per_sample == 3:
		return (struct.unpack('<i', b'\x00' + data[:3])[0] >> 8)
	elif bytes_per_sample == 4:
		return (struct.unpack('<i', data[:4])[0] >> 16)

	return 0


def encode_file(wav_filename, apm_filename, ima_adpcm=False):
	"""
		Converts a PCM .wav file to the Rayman 2 .apm format.

		wav_filename: the input file name
		apm_filename: the output file name
		ima_adpcm: if True, standard IMA-ADPCM encoding is applied;
		           if False, do the custom variant with modified step variable
	"""

	# Open WAV file
	pcm = open(wav_filename, 'rb')

	# Check format
	if pcm.read(4) != b'RIFF':
		print('Error: input file is not a RIFF file.', file=sys.stderr)
		pcm.close()
		return 2
	pcm.seek(4, 1)
	if pcm.read(4) != b'WAVE':
		print('Error: input file is not a WAVE file.', file=sys.stderr)
		pcm.close()
		return 2

	# Locate format subchunk
	chunkHeader = pcm.read(4)
	while chunkHeader and chunkHeader != b'fmt ':
		chunkSize = struct.unpack('<I', pcm.read(4))[0]
		pcm.seek(chunkSize, 1)
		chunkHeader = pcm.read(4)
	if not chunkHeader:
		print('Error: could not find format chunk in wave file.',
				file=sys.stderr)
		pcm.close()
		return 3
	pcm.seek(4, 1)
	audio_format, channel_count, sample_rate = \
			struct.unpack('<HHI', pcm.read(8))[0:3]
	if audio_format != 1:
		print('Error: unsupported audio format ('+str(hex(audio_format))+').',
				file=sys.stderr)
		pcm.close()
		return 4
	if channel_count < 1 or channel_count > 2:
		print('Error: this file has ' + str(channel_count) \
				+ ' channels (only mono and stereo are supported).',
				file=sys.stderr)
		pcm.close()
		return 4
	pcm.seek(6, 1)
	bits_per_sample = struct.unpack('<H', pcm.read(2))[0]
	if bits_per_sample not in [8, 16, 24, 32]:
		print('Unsupported bit depth (' + str(bits_per_sample) \
				+ '); only 8, 16, 24 and 32 are supported.', file=sys.stderr)
		pcm.close()
		return 4
	bytes_per_sample = bits_per_sample >> 3

	# Locate data subchunk
	pcm.seek(12, 0)
	chunkHeader = pcm.read(4)
	while chunkHeader and chunkHeader != b'data':
		chunkSize = struct.unpack('<I', pcm.read(4))[0]
		pcm.seek(chunkSize, 1)
		chunkHeader = pcm.read(4)
	if not chunkHeader:
		print('Error: could not find data chunk in wave file.', file=sys.stderr)
		pcm.close()
		return 3
	data_length = struct.unpack('<I', pcm.read(4))[0]
	data_length = (data_length // (bytes_per_sample * channel_count)) - 1

	# Preinitialized variables
	predictor = [0 for c in range(channel_count)]
	step_index = [0 for c in range(channel_count)]
	sample = [0 for c in range(channel_count)]
	adpcm_data = [] # will contain the following:
	# [initial_predictor, initial_step] * channel_count + [nibbles12,
	# nibbles34, nibbles56, ...]

	# Read the first two PCM samples for each channel
	init_sample = [0 for i in range(channel_count * 2)]
	for i in range(2):
		for c in range(channel_count):
			init_sample[(c * 2) + i] = read_sample_16(pcm, bytes_per_sample)
	# Initialize ADPCM encoder
	for c in range(channel_count):
		predictor[c], step_index[c] = \
				encode_init(init_sample[c * 2], init_sample[(c * 2) + 1])
		adpcm_data.extend([predictor[c], step_index[c]])
	pcm.seek(-channel_count * bytes_per_sample, 1)

	# Start encoding
	remaining_samples = data_length
	nibble = [0 for i in range(channel_count * 2)]
	while remaining_samples > 0:
		for n in range(2): # Two nibbles per byte
			if not (remaining_samples == 0 and n == 1):
				for c in range(channel_count):
					sample[c] = read_sample_16(pcm, bytes_per_sample)
					nibble[c * 2 + n], predictor[c], step_index[c] = \
							encode(sample[c], predictor[c], step_index[c], \
									ima_adpcm)
			else:
				# If the data length is odd, the last nibble is set to null
				for c in range(channel_count):
					nibble[c * 2 + n] = 0
			remaining_samples -= 1

		for c in range(channel_count):
			adpcm_data.append(((nibble[c*2] & 15) << 4) + (nibble[c*2+1] & 15))

	# End of encoding, close file
	pcm.close()

	# Add padding bytes if adpcm_data is less than 4 bytes long, otherwise the
	# header would be screwed up
	minimum_data_length = len(adpcm_data[channel_count*3-1:channel_count*3+3])
	too_short_padding_length = 0
	if minimum_data_length < 4:
		adpcm_data.extend([0 for i in range(minimum_data_length, 4)])
		too_short_padding_length = (4 - minimum_data_length)

	# Create and open APM file
	adp = open(apm_filename, 'wb')

	# APM file rendering...
	# Header
	adp.write(bytearray([0, 0x20])) # Format tag (2)
	adp.write(bytearray(toList(channel_count, 2))) # Channel count (2)
	adp.write(bytearray(toList(sample_rate))) # Sample rate (4)
	adp.write(bytearray(toList(sample_rate*channel_count*2))) # Byte rate (4)
			# (note: byte rate is stored as if the file was uncompressed PCM16)
	adp.write(bytearray([1, 0, 4, 0, 0x50, 0, 0, 0]))
			# Block align (2), Bit rate (2), APM chunk size (4)
	adp.write(bytearray(b'vs12')) # Ubisoft ADPCM version identifier (4)
	file_length = ((data_length+1 if data_length & 1 else data_length) >> 1) \
			* channel_count + 100 + too_short_padding_length
	adp.write(bytearray(toList(file_length))) # Total file length in bytes (4)
	adp.write(bytearray(toList(data_length))) # Sample count (4)
	adp.write(bytearray([0xff, 0xff, 0xff, 0xff])) # Unknown (4)
	adp.write(bytearray(toList(0, 8))) # Unknown (4), Parity flag (4)
	for c in reversed(range(channel_count)):
		adp.write(bytearray(toList(adpcm_data[c*2]))) # Initial PCM value (4)
		adp.write(bytearray(toList(adpcm_data[c*2+1]))) # Initial step index (4)
		adp.write(bytearray(adpcm_data[channel_count*2+c:channel_count*2+c+4]))
				# Audio data start (4)
	adp.write(bytearray([0 for i in range(52 - channel_count*12)])) # Padding

	# Data section
	adp.write(bytearray(b'DATA')) # Data chunk header
	adp.write(bytearray(adpcm_data[channel_count*2:])) # ADPCM data (skipping
	                                                   # the initial values

	# End of rendering, close file
	adp.close()

	return 0



# -----------------------------------------------------------------------------
# Decoder section
# -----------------------------------------------------------------------------

def decode(nibble, stepindex, step, predictor, ima_adpcm=False):
	"""
		Perform one step of the ADPCM decoding algorithm, with given inputs.

		nibble: one ADPCM nibble
		stepindex: the current ADPCM step index
		step: the current ADPCM step
		predictor: the last decoded PCM variable
		ima_adpcm: if False, the lowest 3 bits of the step are cleared

		See http://www.cs.columbia.edu/~hgs/audio/dvi/ and
		https://wiki.multimedia.cx/index.php/IMA_ADPCM for more information.
	"""
	stepindex += index_table[(nibble & 7)]
	diff = step >> 3
	if not ima_adpcm:
		step &= (~7)

	if nibble & 1:
		diff += (step >> 2)
	if nibble & 2:
		diff += (step >> 1)
	if nibble & 4:
		diff += step
	if nibble & 8:
		diff = -diff
	predictor += diff
	if predictor > 32767:
		predictor = 32767
	elif predictor < -32768:
		predictor = -32768
	if stepindex > 88:
		stepindex = 88
	elif stepindex < 0:
		stepindex = 0
	step = step_table[stepindex]

	return predictor, stepindex, step


def decode_file(apm_filename, wav_filename, ima_adpcm=False, \
		preserve_first=False):
	"""
		Converts a Rayman 2 .apm file to 16-bit PCM .wav.

		apm_filename: the input file name
		wav_filename: the output file name
		ima_adpcm: if True, standard IMA-ADPCM decoding is applied;
		           if False, do the custom variant with modified step variable
	"""
	# File buffers
	adp = open(apm_filename, 'rb')

	# Check format
	adp.seek(0x14, 0) # (0 == SEEK_SET)
	if adp.read(4) != b'vs12':
		print('Error: invalid file format (couldn\'t find .apm identifier).',
				file=sys.stderr)
		adp.close()
		return 2
	adp.seek(0, 0)

	# Read audio information
	format_tag, channel_count, sample_rate = \
			struct.unpack('<HHI', adp.read(8))[0:3]
	if format_tag != 0x2000:
		print('Error: input file is not Ubisoft ADPCM (format tag = ' + hex(format_tag)
				+ ', should be 0x2000).', file=sys.stderr)
		adp.close()
		return 2
	if channel_count < 1 or channel_count > 2:
		print('Error: this file has ' + str(channel_count) \
				+ ' channels (only mono and stereo are supported).',
				file=sys.stderr)
		adp.close()
		return 4
	adp.seek(0x1C, 0)
	data_length = struct.unpack('<I', adp.read(4))[0]
	if preserve_first:
		data_length += 1

	pcm = open(wav_filename, 'wb')

	# Wave file rendering...
	# RIFF chunk
	pcm.write(bytearray(b'RIFF'))
	pcm.write(bytearray(toList((data_length * channel_count * 2) + 36)))
	# Format subchunk
	pcm.write(bytearray(b'WAVEfmt '))
	pcm.write(bytearray([16, 0, 0, 0, 1, 0])) # Subchunk size (4), PCM (2)
	pcm.write(bytearray(toList(channel_count, 2))) # Channel count (2)
	pcm.write(bytearray(toList(sample_rate))) # Sample rate (4)
	pcm.write(bytearray(toList(sample_rate*channel_count*2))) # Byte rate (4)
	pcm.write(bytearray(toList((channel_count * 2), 2))) # Block align (2)
	pcm.write(bytearray([16, 0])) # Bits per sample (2)
	# Data subchunk
	pcm.write(bytearray(b'data'))
	pcm.write(bytearray(toList(data_length * channel_count * 2)))

	# Preinitialized ADPCM variables
	nibble = [0 for c in range(channel_count)] # The data to decode
	predictor = [0 for c in range(channel_count)] # The PCM16 data
	step_index = [0 for c in range(channel_count)]
	step = [0 for c in range(channel_count)]

	# Read the initial values
	# (predictor and step index for both channels)
	adp.seek(0x2C, 0)
	for c in reversed(range(channel_count)):
		predictor[c], step_index[c] = struct.unpack('<iI', adp.read(8))[0:2]
		adp.seek(4, 1) # (1 == SEEK_CUR)
		# Initializing steps for predictors
		step[c] = step_table[step_index[c]]

	if preserve_first:
		for c in range(channel_count):
			pcm.write(bytearray(toList(predictor[c],2)))

	# Seek for DATA chunk in apm file
	adp.seek(0x64, 0)

	# Start decoding
	remaining_samples = (data_length - 1) if preserve_first else data_length
	adp_data = adp.read(channel_count)
	while remaining_samples > 0:
		for c in range(channel_count):
			# Read one byte of data per channel
			nibble[c] = struct.unpack('<B', adp_data[c:c+1])[0]

		for n in reversed(range(2)):
			# There are two nibbles per byte ; if the total number of samples
			# in the file is odd, we don't process the last one (which is
			# usually null).
			if not (remaining_samples == 1 and n == 0):
				for c in range(channel_count):
					# Decoding
					predictor[c], step_index[c], step[c] = \
							decode(nibble[c] >> (n << 2), step_index[c], \
							       step[c], predictor[c], ima_adpcm)
					# Writing output
					pcm.write(bytearray(toList(predictor[c], 2)))

		remaining_samples -= 2
		adp_data = adp.read(channel_count)

	# End of decoding, close the files
	pcm.close()
	adp.close()

	return 0



# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def main():
	# Check the command syntax
	err_arg = False
	if len(sys.argv) < 3:
		print('Not enough arguments provided.', file=sys.stderr)
		err_arg = True
	elif sys.argv[1].lower()[0] not in ['e', 'd']:
		print('Unknown action parameter (should be e[i] or d[i1]).',
			file=sys.stderr)
		err_arg = True
	if err_arg:
		print('Usage:')
		print('\tEncoding (finetuned for the game):')
		print('\t\tpython ' + sys.argv[0] + ' e file.wav [file.apm]')
		print('\tEncoding (using standard IMA-ADPCM algorithm):')
		print('\t\tpython ' + sys.argv[0] + ' ei file.wav [file.apm]')
		print('\tDecoding (as done in the game):')
		print('\t\tpython ' + sys.argv[0] + ' d file.apm [file.wav]')
		print('\tDecoding (using standard IMA-ADPCM algorithm):')
		print('\t\tpython ' + sys.argv[0] + ' di file.apm [file.wav]')
		print('\tDecoding (keep first PCM sample in output file):')
		print('\t\tpython ' + sys.argv[0] + ' d1 file.apm [file.wav]')
		return 1

	if not os.path.isfile(sys.argv[2]):
		print('Error: input file does not exist.', file=sys.stderr)
		return 1

	# Generate output file name if not provided
	if len(sys.argv) >= 4:
		output_filename = sys.argv[3]
	else:
		# Give the input file's name to the output file
		output_filename = re.sub('\.[^.]*$', '', sys.argv[2])
		output_filename += '.apm' if sys.argv[1].lower()[0] == 'e' else '.wav'

	if sys.argv[2] == output_filename:
		print('Error: input and output files share the same name.',
				file=sys.stderr)
		return 1

	# Convert the file
	if sys.argv[1].lower()[0] == 'e':
		cursor = 0
		options = sys.argv[1].lower()[1:]
		ima_algorithm = False

		while cursor < len(options):
			if options[cursor] == 'i':
				ima_algorithm = True
			cursor += 1

		return encode_file(sys.argv[2], output_filename, ima_algorithm)

	elif sys.argv[1].lower()[0] == 'd':
		cursor = 0
		options = sys.argv[1].lower()[1:]
		ima_algorithm = False
		preserve_first = False

		while cursor < len(options):
			if options[cursor] == 'i':
				ima_algorithm = True
			elif options[cursor] == '1':
				preserve_first = True
			cursor += 1

		return decode_file(sys.argv[2], output_filename, ima_algorithm, \
				preserve_first)


if __name__ == '__main__':
	main()
