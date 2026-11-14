#!/bin/sh
# Lamina's draft layer for Qwen3.8-Flash-Next (the engine's --mtp folder): four files, 0.8 GB, from
# https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP at a pinned revision, each checked by its SHA-256.
#
#   ./download-draft-layer.sh FOLDER      then in the config's "args":  "--mtp", "FOLDER"
#
# A download that broke off goes on from where it stopped.  HF_ENDPOINT names a mirror.  It needs curl or wget, and
# sha256sum.
set -eu
if [ $# -ne 1 ]; then
  echo "usage: $0 FOLDER" >&2
  echo "Downloads Lamina's draft layer for Qwen3.8-Flash-Next (0.8 GB) into FOLDER; then name the folder in the" >&2
  echo "config's \"args\":  \"--mtp\", \"FOLDER\"" >&2
  exit 2
fi
rev=ff2f70927caafe90ebd1f2ea1c08a9e014a65ce4
url="${HF_ENDPOINT:-https://huggingface.co}/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP/resolve/$rev"
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -L --fail --retry 5 --retry-delay 5 -C - -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -c -O "$2" "$1"; }
else
  echo "curl or wget is needed" >&2
  exit 1
fi
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum is needed (coreutils)" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d ' ' -f 1; }
mkdir -p "$1"
dest=$(cd "$1" && pwd)

get() {   # $1: the file's name, $2: its SHA-256
  file=$dest/$1
  if [ -f "$file" ]; then
    if [ "$(sha "$file")" = "$2" ]; then
      echo "$1: there already, checked"
      return 0
    fi
    echo "$file is not the file expected (its SHA-256 differs): move it away and run this again" >&2
    return 1
  fi
  if [ ! -f "$file.part" ] || [ "$(sha "$file.part")" != "$2" ]; then
    echo "$1 ..."
    if ! fetch "$url/$1" "$file.part"; then
      echo "$1: the download failed. Run this again to go on from where it stopped; if it fails the same way," \
           "delete $file.part first" >&2
      return 1
    fi
    if [ "$(sha "$file.part")" != "$2" ]; then
      echo "$1: the download is not the file expected (its SHA-256 differs): delete $file.part and run this again" >&2
      return 1
    fi
  fi
  mv "$file.part" "$file"
  echo "$1: downloaded, checked"
}

get dense.txt 8773c81ebb0986e37fe94a8a9933e87be48b1fabc6889a0106bcdab179d1c2ac
get draft_vocab.bin 369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25
get dense.bin a46561c5d3e76e78eebb9b54bbf4ae87ba1f99a62cad85562cbb516141cd5e47
get experts.bin 09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f
echo
echo "The draft layer is in $dest"
echo "Name the folder in the config's \"args\":  \"--mtp\", \"$dest\""
