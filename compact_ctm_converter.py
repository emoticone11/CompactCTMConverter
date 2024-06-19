import sys
import numpy as np 
import cv2

ADJ_MAP = {
  0:  {},
  1:  {'e'},
  2:  {'e', 'w'},
  3:  {'w'},
  4:  {'e', 's'},
  5:  {'s', 'w'},
  6:  {'n', 'e', 's'},
  7:  {'e', 's', 'w'},
  8:  {'n', 'ne', 'e', 's', 'w'},
  9:  {'n', 'e', 'se', 's', 'w'},
  10: {'n', 'e', 's', 'sw', 'w', 'nw'},
  11: {'n', 'ne', 'e', 's', 'w', 'nw'},
  12: {'s'},
  13: {'e', 'se', 's'},
  14: {'e', 'se', 's', 'sw', 'w'},
  15: {'s', 'sw', 'w'},
  16: {'n', 'e'},
  17: {'n', 'w'},
  18: {'n', 'e', 'w'},
  19: {'n', 's', 'w'},
  20: {'n', 'e', 's', 'w', 'nw'},
  21: {'n', 'e', 's', 'sw', 'w'},
  22: {'n', 'e', 'se', 's', 'sw', 'w'},
  23: {'n', 'ne', 'e', 'se', 's', 'w'},
  24: {'n', 's'},
  25: {'n', 'ne', 'e', 'se', 's'},
  26: {'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'},
  27: {'n', 's', 'sw', 'w', 'nw'},
  28: {'n', 'e', 'se', 's'},
  29: {'e', 's', 'sw', 'w'},
  30: {'n', 'ne', 'e', 's'},
  31: {'e', 'se', 's', 'w'},
  32: {'n', 'ne', 'e', 's', 'sw', 'w', 'nw'},
  33: {'n', 'ne', 'e', 'se', 's', 'w', 'nw'},
  34: {'n', 'ne', 'e', 's', 'sw', 'w'},
  35: {'n', 'e', 'se', 's', 'w', 'nw'},
  36: {'n'},
  37: {'n', 'ne', 'e'},
  38: {'n', 'ne', 'e', 'w', 'nw'},
  39: {'n', 'w', 'nw'},
  40: {'n', 'ne', 'e', 'w'},
  41: {'n', 's', 'w', 'nw'},
  42: {'n', 'e', 'w', 'nw'},
  43: {'n', 's', 'sw', 'w'},
  44: {'n', 'e', 'se', 's', 'sw', 'w', 'nw'},
  45: {'n', 'ne', 'e', 'se', 's', 'sw', 'w'},
  46: {'n', 'e', 's', 'w'}
}

FULL = 32
HALF = 16
QUARTER = 8


def get_quadrant(texture, n):
  if n == 3:
    return texture[FULL:2*FULL+1, FULL:2*FULL+1, :]
  elif n == 2:
    return texture[FULL:2*FULL+1, :FULL, :]
  elif n == 1:
    return texture[:FULL, FULL:2*FULL+1, :]
  else:
    return texture[:FULL, :FULL, :]


def create_tile(n, source, texture):
  if n == 0:
    return source
  
  tile = source.copy()

  if 'n' in ADJ_MAP[n]:
    tile[:HALF, :, :] = get_quadrant(texture, 1)[:HALF, :, :]
  if 's' in ADJ_MAP[n]:
    tile[HALF:, :, :] = get_quadrant(texture, 1)[HALF:, :, :]
  if 'w' in ADJ_MAP[n]:
    tile[:, :HALF, :] = get_quadrant(texture, 2)[:, :HALF, :]
  if 'e' in ADJ_MAP[n]:
    tile[:, HALF:, :] = get_quadrant(texture, 2)[:, HALF:, :]

  if 'n' in ADJ_MAP[n] and 'w' in ADJ_MAP[n]:
    quadrant = 0 if 'nw' in ADJ_MAP[n] else 3
    tile[:HALF, :HALF, :] = get_quadrant(texture, quadrant)[:HALF, :HALF, :]
  if 'n' in ADJ_MAP[n] and 'e' in ADJ_MAP[n]:
    quadrant = 0 if 'ne' in ADJ_MAP[n] else 3
    tile[:HALF, HALF:, :] = get_quadrant(texture, quadrant)[:HALF, HALF:, :]
  if 's' in ADJ_MAP[n] and 'w' in ADJ_MAP[n]:
    quadrant = 0 if 'sw' in ADJ_MAP[n] else 3
    tile[HALF:, :HALF, :] = get_quadrant(texture, quadrant)[HALF:, :HALF, :]
  if 's' in ADJ_MAP[n] and 'e' in ADJ_MAP[n]:
    quadrant = 0 if 'se' in ADJ_MAP[n] else 3
    tile[HALF:, HALF:, :] = get_quadrant(texture, quadrant)[HALF:, HALF:, :]

  return tile


def convert(source, texture):
  # print(source.shape, texture.shape)
  ctm = np.zeros(shape=(128,384,3), dtype=np.uint8)
  for i in range(4):
    for j in range(12):
      n = 12*i + j
      if n == 47:
        continue
      tile = create_tile(n, source, texture)
      ctm[FULL*i:FULL*(i+1), FULL*j:FULL*(j+1), :] = tile
  return ctm


def read_image(fname):
  return cv2.imread(fname, cv2.IMREAD_UNCHANGED)


def write_image(img, fname):
  cv2.imwrite(fname, img) 


def main(args):
  if len(args) != 2:
    error('Must provide names of compact CTM source file and 2x2 texture file as command line arguments.')
  fname_source = args[0]
  fname_texture = args[1]
  output = convert(read_image(fname_source), read_image(fname_texture))
  fname_out = fname_texture.replace('_2x2.png', '_ctm.png') if '_2x2.png' in fname_texture else fname_texture.replace('.png', '_ctm.png')
  write_image(output, fname_out)


def error(message=None):
  if message:
    print(message)
  exit()


if __name__ == '__main__':
  main(sys.argv[1:])