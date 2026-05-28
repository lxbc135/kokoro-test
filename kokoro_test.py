import argparse
import sys
from kokoro import KPipeline
import soundfile as sf
import torch

# Configure stdout encoding for Windows to support IPA phonetic symbols and unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Language codes dictionary
LANG_CODES = {
    'a': 'American English (US)',
    'b': 'British English (UK)',
    'e': 'Spanish (es)',
    'f': 'French (fr-fr)',
    'h': 'Hindi (hi)',
    'i': 'Italian (it)',
    'p': 'Brazilian Portuguese (pt-br)',
    'j': 'Japanese (ja)',
    'z': 'Mandarin Chinese (zh)',
}

# Fallback voice list if offline
FALLBACK_VOICES = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
    'ef_dora', 'em_alex', 'em_santa',
    'ff_siwis',
    'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
    'if_sara', 'im_nicola',
    'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo',
    'pf_dora', 'pm_alex', 'pm_santa',
    'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
    'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'
]

# Set up command-line argument parser with custom help handling
parser = argparse.ArgumentParser(description="Kokoro TTS Generator CLI", add_help=False)
parser.add_argument("text", nargs="?", type=str, help="The text to synthesize")
parser.add_argument("-l", "--lang_code", type=str, default="a", help="Language code (e.g., 'a' for US English, 'b' for UK English)")
parser.add_argument("-v", "--voice", type=str, default="af_heart", help="Voice to use (e.g., 'af_heart')")
parser.add_argument("-h", "--help", action="store_true", help="Show this help message or list available languages/voices when combined with -l or -v")

# Custom help interception
if '-h' in sys.argv or '--help' in sys.argv:
    has_lang_flag = '-l' in sys.argv or '--lang_code' in sys.argv
    has_voice_flag = '-v' in sys.argv or '--voice' in sys.argv
    
    if has_lang_flag or has_voice_flag:
        if has_lang_flag:
            print("Available Language Codes:")
            for code, name in LANG_CODES.items():
                print(f"  {code} : {name}")
            print()
            
        if has_voice_flag:
            # Determine the language code to filter voices by (checks after -l or --lang_code)
            lang_code = "a"
            for flag in ('-l', '--lang_code'):
                if flag in sys.argv:
                    try:
                        idx = sys.argv.index(flag)
                        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('-'):
                            lang_code = sys.argv[idx + 1]
                    except ValueError:
                        pass
            
            # Fetch voices (dynamic with fallback)
            voices = FALLBACK_VOICES
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files('hexgrad/Kokoro-82M')
                voices = [f.split('/')[-1].replace('.pt', '') for f in files if f.startswith('voices/') and f.endswith('.pt')]
            except Exception:
                pass
                
            # Filter voices for the language code (prefixes, e.g., 'af_' or 'am_')
            prefix_female = f"{lang_code}f_"
            prefix_male = f"{lang_code}m_"
            filtered_voices = [v for v in voices if v.startswith(prefix_female) or v.startswith(prefix_male)]
            
            lang_name = LANG_CODES.get(lang_code, lang_code)
            print(f"Available Voices for Language Code '{lang_code}' ({lang_name}):")
            if filtered_voices:
                for voice in sorted(filtered_voices):
                    gender = "Female" if voice.startswith(prefix_female) else "Male"
                    print(f"  {voice} ({gender})")
            else:
                print(f"  No voices found matching language code '{lang_code}'.")
            print()
            
        sys.exit(0)
    else:
        # Standard general help
        parser.print_help()
        sys.exit(0)

args = parser.parse_args()

# Validate that text was provided
if not args.text:
    print("Error: The text to synthesize is required as the first argument.", file=sys.stderr)
    print()
    parser.print_help()
    sys.exit(1)

# Initialize the pipeline
pipeline = KPipeline(lang_code=args.lang_code)

# Generate audio
generator = pipeline(
    args.text,
    voice=args.voice,
    speed=1,
    split_pattern=r'\n+'
)

# Loop and save audio files
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Segment {i}:")
    print(f"  Graphemes: {gs}")
    print(f"  Phonemes: {ps}")
    output_filename = f"{i}.wav"
    sf.write(output_filename, audio, 24000)
    print(f"  Saved to: {output_filename}")
