#!/usr/bin/env python

"""
    Ray2Get 2.0
    by Palorifgrodbierzrt 2018
    [ver. 2026-01-20]


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
the game does, and another one which preserves the first sample in the output
file (in case you're a purist like me :p).


Usage:
  Encoding, for proper play in game:  python ray2get.py -e file.wav [file.apm]
  Encoding, regular IMA-ADPCM:        python ray2get.py -ei file.wav [file.apm]

  Decoding, as the game would:        python ray2get.py -d file.apm [file.wav]
  Decoding, regular IMA-ADPCM:        python ray2get.py -di file.apm [file.wav]
  Decoding, keep 1st sample:          python ray2get.py -d1 file.apm [file.wav]
  Display a progress bar:             python ray2get.py -v file.wav

Flags 'i', '1' and 'v' are optional and can be used interchangeably.
The action flag (-e or -d) is also optional; if not provided, the script will
attempt to guess it from the input file extension (.wav -> encode,
.apm -> decode).
The last parameter (output file name) is optional. If not provided, the output
file name will be the same as the input file name, with the extension replaced
with .apm (or appended if the input file has no extension).


Special thanks to the folks from the Rayman Pirate Community, deton24 for
(unknowingly) pointing some of my bugs out, imaginaryPineapple (from the
OpenRayman project) for the complementary info about the APM file format.
"""


import argparse, os, struct, sys


# -----------------------------------------------------------------------------
# Common data
# -----------------------------------------------------------------------------

# The IMA index table
index_table = [-1, -1, -1, -1, 2, 4, 6, 8]

# The IMA step table
step_table = [
    0x0007, 0x0008, 0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x000e,
    0x0010, 0x0011, 0x0013, 0x0015, 0x0017, 0x0019, 0x001c, 0x001f,
    0x0022, 0x0025, 0x0029, 0x002d, 0x0032, 0x0037, 0x003c, 0x0042,
    0x0049, 0x0050, 0x0058, 0x0061, 0x006b, 0x0076, 0x0082, 0x008f,
    0x009d, 0x00ad, 0x00be, 0x00d1, 0x00e6, 0x00fd, 0x0117, 0x0133,
    0x0151, 0x0173, 0x0198, 0x01c1, 0x01ee, 0x0220, 0x0256, 0x0292,
    0x02d4, 0x031c, 0x036c, 0x03c3, 0x0424, 0x048e, 0x0502, 0x0583,
    0x0610, 0x06ab, 0x0756, 0x0812, 0x08e0, 0x09c3, 0x0abd, 0x0bd0,
    0x0cff, 0x0e4c, 0x0fba, 0x114c, 0x1307, 0x14ee, 0x1706, 0x1954,
    0x1bdc, 0x1ea5, 0x21b6, 0x2515, 0x28ca, 0x2cdf, 0x315b, 0x364b,
    0x3bb9, 0x41b2, 0x4844, 0x4f7e, 0x5771, 0x602f, 0x69ce, 0x7462,
    0x7fff
]


def print_progress(current, total, width=40):
    """
        Print a progress bar to the console.
    """
    percent = (current / total) if total > 0 else 1.0
    filled_length = int(width * percent)
    bar = '=' * filled_length + '>' + ' ' * (width - filled_length - 1)
    if filled_length >= width:
        bar = '=' * width
    elif filled_length < 0:
        bar = ' ' * width
    sys.stdout.write(f'\r[{bar}] {percent*100:5.1f}%')
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write('\n')


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


def read_sample(file_handler, bytes_per_sample):
    """
        Reads a PCM sample from a file, and converts it to 16 bits.

        file_handler: a file handler to .read() the bytes from
        bytes_per_sample: an integer between 1 and 4 (only 8, 16, 24 and 32-bit
                          files are supported)
    """
    data = file_handler.read(bytes_per_sample)
    if data == b'':
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


def encode_file(wav_filename, apm_filename, ima_adpcm=False, verbose=False):
    """
        Converts a PCM .wav file to the Rayman 2 .apm format.

        wav_filename: the input file name
        apm_filename: the output file name
        ima_adpcm: if True, uses regular IMA-ADPCM encoding;
                   if False, uses the custom variant with modified step
                   variable
        verbose: if True, displays a progress bar
    """

    # Open WAV file
    with open(wav_filename, 'rb') as pcm:
        # Check format
        if pcm.read(4) != b'RIFF':
            print('Error: input file is not a RIFF file', file=sys.stderr)
            return 2
        pcm.seek(4, 1)
        if pcm.read(4) != b'WAVE':
            print('Error: input file is not a WAVE file', file=sys.stderr)
            return 2

        # Locate format subchunk
        chunkHeader = pcm.read(4)
        while chunkHeader and chunkHeader != b'fmt ':
            chunkSize = struct.unpack('<I', pcm.read(4))[0]
            pcm.seek(chunkSize, 1)
            chunkHeader = pcm.read(4)
        if not chunkHeader:
            print('Error: could not find format chunk in wave file',
                file=sys.stderr)
            return 3
        pcm.seek(4, 1)
        audio_format, channel_count, sample_rate = \
            struct.unpack('<HHI', pcm.read(8))[0:3]
        if audio_format != 1:
            print(f'Error: unsupported audio format (0x{audio_format:04x})',
                file=sys.stderr)
            return 4
        if channel_count < 1 or channel_count > 2:
            print(f'Error: this file has {channel_count} channels (only mono '
                  'and stereo are supported)', file=sys.stderr)
            return 4
        pcm.seek(6, 1)
        bits_per_sample = struct.unpack('<H', pcm.read(2))[0]
        if bits_per_sample not in [8, 16, 24, 32]:
            print(f'Unsupported bit depth ({bits_per_sample}); only 8, 16, 24 '
                  'and 32 are supported', file=sys.stderr)
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
            print('Error: could not find data chunk in wave file',
                file=sys.stderr)
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
                init_sample[(c * 2) + i] = read_sample(pcm, bytes_per_sample)
        # Initialize ADPCM encoder
        for c in range(channel_count):
            predictor[c], step_index[c] = \
                encode_init(init_sample[c * 2], init_sample[(c * 2) + 1])
            adpcm_data.extend([predictor[c], step_index[c]])
        pcm.seek(-channel_count * bytes_per_sample, 1)

        # Start encoding
        total_samples = data_length
        remaining_samples = data_length
        nibble = [0 for i in range(channel_count * 2)]
        while remaining_samples > 0:
            if verbose and (total_samples - remaining_samples) % 1024 == 0:
                print_progress(total_samples - remaining_samples,
                               total_samples)
            for n in range(2): # Two nibbles per byte
                if not (remaining_samples == 0 and n == 1):
                    for c in range(channel_count):
                        sample[c] = read_sample(pcm, bytes_per_sample)
                        nibble[c * 2 + n], predictor[c], step_index[c] = \
                            encode(sample[c], predictor[c], step_index[c],
                                ima_adpcm)
                else:
                    # If the data length is odd, the last nibble is set to null
                    for c in range(channel_count):
                        nibble[c * 2 + n] = 0
                remaining_samples -= 1

            for c in range(channel_count):
                adpcm_data.append(((nibble[c*2] & 15) << 4)
                                  + (nibble[c*2+1] & 15))
 
        if verbose:
            print_progress(total_samples, total_samples)

    # Add padding bytes if adpcm_data is less than 4 bytes long, otherwise the
    # header would be screwed up
    minimum_data_length = len(adpcm_data[channel_count*3-1:channel_count*3+3])
    too_short_padding_length = 0
    if minimum_data_length < 4:
        adpcm_data.extend([0 for i in range(minimum_data_length, 4)])
        too_short_padding_length = (4 - minimum_data_length)

    # Create and open APM file
    file_length = \
        ((data_length + 1 if data_length & 1 else data_length) >> 1) \
        * channel_count + 100 + too_short_padding_length

    header = struct.pack('<HHI IHH I 4s IIIII',
        0x2000,                           # Format tag (Ubisoft ADPCM)
        channel_count,                    # Channel count
        sample_rate,                      # Sample rate
        sample_rate * channel_count * 2,  # Byte rate (approx. PCM16)
        1,                                # Block align
        4,                                # Bits per sample
        0x50,                             # APM chunk size
        b'vs12',                          # Version identifier
        file_length,                      # Total file length
        data_length,                      # Sample count
        0xffffffff,                       # Unknown
        0,                                # Unknown
        0                                 # Parity flag placeholder
    )

    # Initial values for each channel (interleaved, right channel first)
    initial_values = b''
    for c in reversed(range(channel_count)):
        initial_values += struct.pack('<iI 4s',
            adpcm_data[c*2],              # Initial PCM value
            adpcm_data[c*2+1],            # Initial step index
            bytearray(adpcm_data[channel_count*2+c:channel_count*2+c+4])
        )

    # Padding for the 80-byte APM block (52 bytes total for channel info)
    padding = b'\x00' * (52 - channel_count * 12)

    with open(apm_filename, 'wb') as adp:
        adp.write(header)
        adp.write(initial_values)
        adp.write(padding)
        adp.write(b'DATA')                # Data chunk header
        adp.write(bytearray(adpcm_data[channel_count*2:]))

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


def decode_file(apm_filename, wav_filename, ima_adpcm=False,
    preserve_first=False, verbose=False):
    """
        Converts a Rayman 2 .apm file to 16-bit PCM .wav.

        apm_filename: the input file name
        wav_filename: the output file name
        ima_adpcm: if True, uses regular IMA-ADPCM decoding;
                   if False, uses the custom variant with modified step
                   variable
        preserve_first: if True, the first PCM sample is not discarded
        verbose: if True, displays a progress bar
    """

    # File buffers
    with open(apm_filename, 'rb') as adp:
        # Check format
        adp.seek(0x14, 0) # (0 == SEEK_SET)
        if adp.read(4) != b'vs12':
            print('Error: invalid file format (couldn\'t find .apm'
                  'identifier)', file=sys.stderr)
            return 2
        adp.seek(0, 0)

        # Read audio information
        format_tag, channel_count, sample_rate = \
            struct.unpack('<HHI', adp.read(8))[0:3]
        if format_tag != 0x2000:
            print(f'Error: input file is not Ubisoft ADPCM (format tag = '
                  f'0x{format_tag:04x}, should be 0x2000)', file=sys.stderr)
            return 2
        if channel_count < 1 or channel_count > 2:
            print(f'Error: this file has {channel_count} channels (only mono '
                  'and stereo are supported)', file=sys.stderr)
            return 4
        adp.seek(0x1C, 0)
        data_length = struct.unpack('<I', adp.read(4))[0]
        if preserve_first:
            data_length += 1

        with open(wav_filename, 'wb') as pcm:
            # Wave file rendering...
            # RIFF chunk
            pcm.write(b'RIFF')
            pcm.write(((data_length * channel_count * 2) + 36)
                      .to_bytes(4, 'little'))
            # Format subchunk
            pcm.write(b'WAVEfmt ')
            pcm.write((16).to_bytes(4, 'little'))  # Subchunk size
            pcm.write((1).to_bytes(2, 'little'))   # PCM
            pcm.write(channel_count.to_bytes(2, 'little'))  # Channel count
            pcm.write(sample_rate.to_bytes(4, 'little'))  # Sample rate
            pcm.write((sample_rate * channel_count * 2)
                      .to_bytes(4, 'little'))  # Byte rate
            pcm.write((channel_count * 2).to_bytes(2, 'little'))  # Block align
            pcm.write((16).to_bytes(2, 'little'))  # Bits per sample
            # Data subchunk
            pcm.write(b'data')
            pcm.write((data_length * channel_count * 2).to_bytes(4, 'little'))

            # Preinitialized ADPCM variables
            nibble = [0 for c in range(channel_count)]  # The data to decode
            predictor = [0 for c in range(channel_count)]  # The PCM16 data
            step_index = [0 for c in range(channel_count)]
            step = [0 for c in range(channel_count)]

            # Read the initial values
            # (predictor and step index for both channels)
            adp.seek(0x2C, 0)
            for c in reversed(range(channel_count)):
                predictor[c], step_index[c] = \
                    struct.unpack('<iI', adp.read(8))[0:2]
                adp.seek(4, 1)  # (1 == SEEK_CUR)
                # Initializing steps for predictors
                step[c] = step_table[step_index[c]]

            if preserve_first:
                for c in range(channel_count):
                    pcm.write(predictor[c].to_bytes(2, 'little', signed=True))

            # Seek for DATA chunk in apm file
            adp.seek(0x64, 0)

            # Start decoding
            total_samples = (data_length - 1) if preserve_first \
                                              else data_length
            remaining_samples = total_samples
            adp_data = adp.read(channel_count)
            while remaining_samples > 0:
                if verbose and (total_samples - remaining_samples) % 1024 == 0:
                    print_progress(total_samples - remaining_samples,
                                   total_samples)
                for c in range(channel_count):
                    # Read one byte of data per channel
                    nibble[c] = struct.unpack('<B', adp_data[c:c+1])[0]

                for n in reversed(range(2)):
                    # There are two nibbles per byte ; if the total number of
                    # samples in the file is odd, we don't process the last one
                    # (which is usually null).
                    if not (remaining_samples == 1 and n == 0):
                        for c in range(channel_count):
                            # Decoding
                            predictor[c], step_index[c], step[c] = \
                                decode(nibble[c] >> (n << 2), step_index[c],
                                       step[c], predictor[c], ima_adpcm)
                            # Writing output
                            pcm.write(predictor[c].to_bytes(2, 'little',
                                      signed=True))

                remaining_samples -= 2
                adp_data = adp.read(channel_count)
 
            if verbose:
                print_progress(total_samples, total_samples)

    return 0


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Convert .wav files to .apm '
                                     'and vice versa.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-e', '--encode', action='store_true',
                        help='encode a .wav file to .apm using Rayman 2\'s '
                        'algorithm variant (default)')
    group.add_argument('-d', '--decode', action='store_true',
                        help='decode an .apm file to .wav using Rayman 2\'s '
                        'algorithm variant (default)')
    parser.add_argument('-i', '--ima-adpcm', action='store_true',
                        help='use the regular IMA-ADPCM algorithm')
    parser.add_argument('-1', '--preserve-first', action='store_true',
                        help='keep the first PCM sample in the output file '
                        '(only when decoding)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force overwriting the output file if it '
                        'already exists')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='display a progress bar')
    parser.add_argument('input_file', type=str,
                        help='input file')
    parser.add_argument('output_file', type=str, nargs='?',
                        help='output file')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print('Error: input file does not exist', file=sys.stderr)
        return 1

    # If neither encode nor decode is explicitly requested, try to guess
    # from extension
    if not args.encode and not args.decode:
        extension = os.path.splitext(args.input_file)[1].lower()
        if extension == '.wav':
            args.encode = True
        elif extension == '.apm':
            args.decode = True
        else:
            print('Error: could not determine action (encode/decode) from '
                  'extension. Please use -e or -d flags.', file=sys.stderr)
            return 1

    # Generate output file name if not provided
    if args.output_file:
        output_filename = args.output_file
    else:
        # Give the input file's name to the output file
        output_filename = os.path.splitext(args.input_file)[0]
        output_filename += '.apm' if args.encode else '.wav'

    if args.input_file == output_filename:
        print('Error: input and output files share the same name',
                file=sys.stderr)
        return 1

    # Check whether the output file already exists
    if os.path.isfile(output_filename) and not args.force:
        print('Error: output file already exists', file=sys.stderr)
        return 1

    # Convert the file
    if args.encode:
        return encode_file(args.input_file, output_filename, args.ima_adpcm,
                           args.verbose)

    elif args.decode:
        return decode_file(args.input_file, output_filename, args.ima_adpcm,
                           args.preserve_first, args.verbose)

    return 0


if __name__ == '__main__':
    main()
