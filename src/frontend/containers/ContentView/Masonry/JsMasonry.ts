/**
 * Pure-JS port of the masonry layout in `wasm/masonry/src/layout.rs`.
 *
 * The WebAssembly layout shares memory with a worker, which needs a cross-origin isolated
 * renderer (`self.crossOriginIsolated`). When the app runs without that isolation the
 * worker cannot be created, so this fallback keeps the gallery laid out and usable.
 */
export const JsMasonryType = {
  Vertical: 0,
  Horizontal: 1,
  Grid: 2,
} as const;

/** Matches the numeric values of `MasonryType` from wasm/packages/masonry. */
export type JsMasonryTypeValue = (typeof JsMasonryType)[keyof typeof JsMasonryType] | number;

export type JsTransform = Readonly<[width: number, height: number, top: number, left: number]>;

/** Very wide/narrow images are cropped so they are at most ~3.33x longer than wide (or vice versa). */
const MIN_ASPECT_RATIO = 100 / 3;

export class JsMasonry {
  #transforms: [number, number, number, number][] = [];
  #dims: { w: number; h: number }[] = [];
  #numItems = 0;
  #thumbnailSize = 0;
  #padding = 0;

  constructor(numItems: number) {
    this.resize(numItems);
  }

  resize(numItems: number): void {
    this.#numItems = numItems;
    while (this.#transforms.length < numItems) {
      this.#transforms.push([0, 0, 0, 0]);
      this.#dims.push({ w: 1, h: 1 });
    }
  }

  setDimension(index: number, width: number, height: number): void {
    if (index < 0 || index >= this.#dims.length) {
      return;
    }
    this.#dims[index] = { w: Math.max(1, width), h: Math.max(1, height) };
  }

  setThumbnailSize(size: number): void {
    this.#thumbnailSize = Math.min(size, Math.floor(65535 / 100));
  }

  setPadding(padding: number): void {
    this.#padding = padding;
  }

  /** Aspect ratio clamped like `correct_aspect_ratio()` in the Rust implementation. */
  #ratio(index: number): { w: number; h: number } {
    const { w, h } = this.#dims[index];
    if (w > h) {
      const height = Math.max(1, Math.floor((100 * h) / w));
      return 100 / height > MIN_ASPECT_RATIO
        ? { w: 100, h: MIN_ASPECT_RATIO }
        : { w: 100, h: height };
    }
    const width = Math.max(1, Math.floor((100 * w) / h));
    return 100 / width > MIN_ASPECT_RATIO ? { w: MIN_ASPECT_RATIO, h: 100 } : { w: width, h: 100 };
  }

  #heightForWidth(index: number, width: number): number {
    const ratio = this.#ratio(index);
    return Math.floor((width * ratio.h) / ratio.w);
  }

  #widthForHeight(index: number, height: number): number {
    const ratio = this.#ratio(index);
    return Math.floor((height * ratio.w) / ratio.h);
  }

  compute(containerWidth: number, type: JsMasonryTypeValue): number {
    if (this.#numItems === 0 || this.#thumbnailSize === 0) {
      return 0;
    }
    switch (type) {
      case JsMasonryType.Horizontal:
        return this.#computeHorizontal(containerWidth);
      case JsMasonryType.Grid:
        return this.#computeGrid(containerWidth);
      default:
        return this.#computeVertical(containerWidth);
    }
  }

  #columns(containerWidth: number): { count: number; width: number } {
    const width = Math.max(containerWidth, this.#thumbnailSize);
    const count = Math.max(1, Math.floor(width / this.#thumbnailSize));
    return { count, width: Math.floor(width / count) };
  }

  #computeVertical(containerWidth: number): number {
    const { count, width: columnWidth } = this.#columns(containerWidth);
    const itemWidth = columnWidth - this.#padding;
    const heights = new Array<number>(count).fill(0);

    for (let i = 0; i < this.#numItems; i++) {
      const height = this.#heightForWidth(i, itemWidth);
      let shortest = 0;
      for (let c = 1; c < count; c++) {
        if (heights[c] < heights[shortest]) {
          shortest = c;
        }
      }
      const top = heights[shortest];
      this.#transforms[i] = [itemWidth, height, top, shortest * columnWidth];
      heights[shortest] = top + height + this.#padding;
    }
    return Math.max(...heights);
  }

  #computeGrid(containerWidth: number): number {
    const { count, width: rowHeight } = this.#columns(containerWidth);
    const itemSize = rowHeight - this.#padding;
    for (let i = 0; i < this.#numItems; i++) {
      const row = Math.floor(i / count);
      const col = i % count;
      this.#transforms[i] = [itemSize, itemSize, row * rowHeight, col * rowHeight];
    }
    return Math.ceil(this.#numItems / count) * rowHeight;
  }

  #computeHorizontal(containerWidth: number): number {
    const maxWidth = Math.max(containerWidth, this.#thumbnailSize);
    const rowHeight = this.#thumbnailSize;
    let top = 0;
    let rowWidth = 0;
    let rowStart = 0;

    const finishRow = (endExclusive: number) => {
      if (rowWidth <= maxWidth) {
        return;
      }
      const factor = maxWidth / rowWidth;
      let left = 0;
      for (let i = rowStart; i < endExclusive; i++) {
        const [w, h] = this.#transforms[i];
        const scaledW = Math.floor(w * factor);
        const scaledH = Math.floor(h * factor);
        this.#transforms[i] = [scaledW, scaledH, top, left];
        left += scaledW + this.#padding;
      }
    };

    for (let i = 0; i < this.#numItems; i++) {
      const width = this.#widthForHeight(i, rowHeight);
      this.#transforms[i] = [width, rowHeight, top, rowWidth];
      rowWidth += width + this.#padding;
      if (rowWidth > maxWidth) {
        finishRow(i + 1);
        rowWidth = 0;
        rowStart = i + 1;
        top += rowHeight + this.#padding;
      }
    }
    if (rowWidth > 0) {
      top += rowHeight + this.#padding;
    }
    return top;
  }

  getTransform(index: number): JsTransform {
    // Items are sized up front, but guard against an out-of-range index anyway.
    return (this.#transforms[index] || [0, 0, 0, 0]) as unknown as JsTransform;
  }
}
