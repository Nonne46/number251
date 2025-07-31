import hashlib
import os

import rarfile

RAR_EXT = ".rar"
DMM_EXT = ".dmm"
OUT_DIR = "extracted_dmm_files"


def find_rar_archives(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(RAR_EXT):
                yield os.path.join(dirpath, f)


def extract_dmm_files(rar_path):
    try:
        with rarfile.RarFile(rar_path) as rf:
            for info in rf.infolist():
                if info.filename.lower().endswith(DMM_EXT):
                    data = rf.read(info)

                    # CRC in rarfile is 32-bit int; we convert it to hex
                    crc_hex = f"{info.CRC & 0xFFFFFFFF:08X}"

                    archive_name = os.path.splitext(os.path.basename(rar_path))[0]
                    original_name = os.path.basename(info.filename).replace(".dmm", "")
                    new_name = f"{archive_name}_{original_name}_{crc_hex}.dmm"
                    out_path = os.path.join(OUT_DIR, new_name)

                    os.makedirs(OUT_DIR, exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    print(f"Extracted: {out_path}")
    except rarfile.Error as e:
        print(f"Failed to read {rar_path}: {e}")


def main(start_dir):
    for rar_path in find_rar_archives(start_dir):
        extract_dmm_files(rar_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_dmm_from_rar.py /path/to/search")
    else:
        main(sys.argv[1])
