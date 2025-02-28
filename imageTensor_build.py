import numpy as np
import pysam
import cv2
import argparse
import os
import json

BASE_ENCODING = {
    "A": [255, 0, 0],  # Red
    "T": [0, 255, 0],  # Green
    "C": [0, 0, 255],  # Blue
    "G": [255, 255, 0],  # Yellow
    "N": [0, 0, 0]  # Black for unknown bases
}


def encode_base(base):
    """Convert a base to RGB encoding."""
    return BASE_ENCODING.get(base.upper(), [0, 0, 0])  # Default to black if unknown


def write_metadata(metadata, output_dir):
    """Write metadata to JSON file."""
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)


def generate_pileup_image(vcf_file, bam_file, fasta_file, output_dir, chromosome=None, flanking_region=100,
                          max_reads=100):
    """Generate image-like pileup representations for CNN models with .vcf.gz support and reference FASTA."""

    os.makedirs(output_dir, exist_ok=True)
    vcf = pysam.VariantFile(vcf_file, "r")
    bam = pysam.AlignmentFile(bam_file, "rb")
    fasta = pysam.FastaFile(fasta_file) if fasta_file else None

    metadata = []
    variant_count = 0
    processed_count = 0

    for record in vcf:
        if chromosome and record.chrom != chromosome:
            continue

        chrom = record.chrom
        pos = record.pos
        ref = record.ref
        alt = ",".join(map(str, record.alts))

        genotype = "NA"
        if "GT" in record.format:
            sample = next(iter(record.samples.values()))
            genotype = "|".join(map(str, sample["GT"]))

        start = max(0, pos - flanking_region)
        end = pos + flanking_region
        reads = [read for read in bam.fetch(chrom, start, end) if not read.is_unmapped]

        if len(reads) == 0:
            continue

        variant_count += 1
        processed_count += 1
        reads = reads[:max_reads]

        img = np.zeros((max_reads + 1, 2 * flanking_region + 1, 3), dtype=np.uint8)

        if fasta:
            ref_seq = fasta.fetch(chrom, start, end)
            for i, base in enumerate(ref_seq):
                img[0, i] = encode_base(base)

        for i, read in enumerate(reads, start=1):
            seq = read.query_alignment_sequence
            aligned_positions = read.get_aligned_pairs(matches_only=True)
            for query_pos, ref_pos in aligned_positions:
                if ref_pos is not None and query_pos is not None:
                    if start <= ref_pos <= end:
                        base = seq[query_pos] if query_pos < len(seq) else "N"
                        img[i, ref_pos - start] = encode_base(base)

        sanitized_alt = alt.replace(",", "_")
        variant_id = f"{chrom}_{pos}_{ref}_{sanitized_alt}_GT{genotype}"
        image_path = os.path.join(output_dir, f"{variant_id}.png")
        npy_path = os.path.join(output_dir, f"{variant_id}.npy")

        cv2.imwrite(image_path, img)
        np.save(npy_path, img)

        metadata.append({
            "chrom": chrom,
            "position": pos,
            "ref": ref,
            "alt": alt,
            "genotype": genotype,
            "image_path": image_path,
            "npy_path": npy_path
        })

        if processed_count % 10000 == 0:
            write_metadata(metadata, output_dir)
            print(f"Saved metadata.json after {processed_count} processed variants")

        print(f"Processed variant {processed_count}: {chrom}:{pos}, Genotype={genotype}")

    write_metadata(metadata, output_dir)
    print(f"Final metadata.json written. Total variants processed: {processed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pileup images for CNN models with reference sequence support.")
    parser.add_argument("vcf", help="Input compressed VCF (.vcf.gz) file")
    parser.add_argument("bam", help="Input BAM file")
    parser.add_argument("fasta", help="Reference genome FASTA file")
    parser.add_argument("output_dir", help="Output directory for images and npy files")
    parser.add_argument("--chromosome", type=str, default="chr20", help="Specify chromosome to process (e.g., 'chr20')")
    parser.add_argument("--flanking", type=int, default=100, help="Flanking region size in bp (default: 100)")
    parser.add_argument("--max_reads", type=int, default=100, help="Max reads per pileup (default: 100)")

    args = parser.parse_args()
    generate_pileup_image(args.vcf, args.bam, args.fasta, args.output_dir, args.chromosome, args.flanking,
                          args.max_reads)