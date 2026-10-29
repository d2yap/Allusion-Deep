import { runInAction } from 'mobx';

import { ClientFile } from '../../../entities/File';
// Force Webpack to include worker and WASM file in the build folder!
import { MasonryType, MasonryWorker, default as init } from 'wasm/packages/masonry';
import { JsMasonry } from './JsMasonry';
import { ITransform, Layouter } from './layout-helpers';

export interface MasonryOptions {
  type: MasonryType;
  thumbSize: number;
  padding: number;
}

const defaultOpts: MasonryOptions = {
  type: MasonryType.Vertical,
  thumbSize: 300,
  padding: 8,
};

export class MasonryWorkerAdapter implements Layouter {
  private worker?: MasonryWorker;
  private memory?: WebAssembly.Memory;
  /** Used when the renderer is not cross-origin isolated, so the wasm worker cannot start. */
  private fallback?: JsMasonry;
  private dimensions: [number, number][] = [];

  private prevNumImgs: number = 0;

  private get needsFallback(): boolean {
    return typeof self !== 'undefined' && self.crossOriginIsolated === false;
  }

  async initialize(numItems: number) {
    this.prevNumImgs = numItems;

    if (this.memory !== undefined && this.worker !== undefined) {
      return;
    }
    if (this.fallback !== undefined) {
      this.fallback.resize(numItems);
      return;
    }

    if (this.needsFallback) {
      console.debug('masonry: no cross-origin isolation, using JS layout fallback');
      this.fallback = new JsMasonry(numItems);
      return;
    }

    try {
      console.debug('initializing masonry worker...');
      const wasm = await init();
      this.memory = wasm.memory;

      const worker = new Worker(new URL('wasm/packages/masonry/worker.js', import.meta.url), {
        type: 'module',
      });
      worker.postMessage(this.memory);
      this.worker = new MasonryWorker(numItems);
    } catch (err) {
      // SharedArrayBuffer transfer needs isolation; keep the gallery working regardless.
      console.error('Could not start the masonry worker, falling back to JS layout', err);
      this.memory = undefined;
      this.worker = undefined;
      this.fallback = new JsMasonry(numItems);
      this.dimensions.forEach(([w, h], i) => this.fallback!.setDimension(i, w, h));
    }
  }

  async compute(
    imgs: ClientFile[],
    numImgs: number,
    containerWidth: number,
    opts: Partial<MasonryOptions>,
  ): Promise<number | undefined> {
    if (this.fallback !== undefined) {
      this.fallback.resize(numImgs);
      this.dimensions = [];
      for (let i = 0; i < imgs.length; i++) {
        this.dimensions[i] = [imgs[i].width, imgs[i].height];
        this.fallback.setDimension(i, imgs[i].width, imgs[i].height);
      }
      this.fallback.setThumbnailSize(opts.thumbSize || defaultOpts.thumbSize);
      this.fallback.setPadding(opts.padding ?? defaultOpts.padding);
      return this.fallback.compute(containerWidth, opts.type ?? defaultOpts.type);
    }

    const worker = this.worker;
    if (worker === undefined) {
      return Promise.reject('Worker is uninitialized.');
    }

    if (this.prevNumImgs !== numImgs) {
      worker.resize(numImgs);
    }

    this.prevNumImgs = numImgs;
    runInAction(() => {
      for (let i = 0; i < imgs.length; i++) {
        worker.set_dimension(i, imgs[i].width, imgs[i].height);
      }
    });

    await worker.compute(
      containerWidth,
      opts.type || defaultOpts.type,
      opts.thumbSize || defaultOpts.thumbSize,
      opts.padding || defaultOpts.padding,
    );
    return worker.get_height();
  }

  async recompute(
    containerWidth: number,
    opts: Partial<MasonryOptions>,
  ): Promise<number | undefined> {
    if (this.fallback !== undefined) {
      this.fallback.setThumbnailSize(opts.thumbSize || defaultOpts.thumbSize);
      this.fallback.setPadding(opts.padding ?? defaultOpts.padding);
      return this.fallback.compute(containerWidth, opts.type ?? defaultOpts.type);
    }
    if (this.worker === undefined) {
      return Promise.reject('Worker is uninitialized.');
    }
    await this.worker.compute(
      containerWidth,
      opts.type || defaultOpts.type,
      opts.thumbSize || defaultOpts.thumbSize,
      opts.padding || defaultOpts.padding,
    );
    return this.worker.get_height();
  }

  // This method will be available in the custom VirtualizedRenderer component as layout.getItemLayout
  getTransform(index: number): ITransform {
    if (this.fallback !== undefined) {
      return this.fallback.getTransform(index);
    }
    if (this.worker === undefined || this.memory === undefined) {
      throw new Error('Worker is uninitialized.');
    }
    const ptr = this.worker.get_transform(index);
    return new Uint32Array(this.memory.buffer, ptr, 4) as unknown as ITransform;
  }
}
