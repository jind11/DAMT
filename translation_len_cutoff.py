import sys

src_file = sys.argv[1]
tgt_file = sys.argv[2]
min_len = int(sys.argv[3])
max_len = int(sys.argv[4])
out_src_file = sys.argv[5]
out_tgt_file = sys.argv[6]

src = open(src_file).readlines()
tgt = open(tgt_file).readlines()

with open(out_src_file, 'w') as src_file_writer, open(out_tgt_file, 'w') as tgt_file_writer:
    for src_line, tgt_line in zip(src, tgt):
        src_line_len = len(src_line.split())
        tgt_line_len = len(tgt_line.split())
        if src_line_len >= min_len and src_line_len <= max_len and tgt_line_len >= min_len and tgt_line_len <= max_len:
            src_file_writer.write(src_line)
            tgt_file_writer.write(tgt_line)
