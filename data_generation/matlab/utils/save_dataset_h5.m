function save_dataset_h5(filepath, signals, labels, snr_db, ...
                          pulse_widths_us, cfg)
% SAVE_DATASET_H5  Write the synthetic radar dataset to HDF5.
%
% Schema (matching docs/decisions.md, 2026-05-04 main loop entry):
%
%   dataset.h5
%   ├── /signals        (N, L) complex64    [clean baseband signals]
%   ├── /labels         (N,)   uint8        [class index 0..7]
%   ├── /snr_db         (N,)   float32      [intended SNR per sample]
%   ├── /pulse_widths_us(N,)   float32      [active pulse width in us]
%   ├── /class_names    (8,)   string       [human-readable labels]
%   └── attributes (root):
%         sample_rate_hz, signal_length, master_seed, generation_date,
%         dataset_version, samples_per_class, num_classes
%
% Inputs
% ------
%   filepath        : output path, e.g. 'synthetic_samples/dataset.h5'
%   signals         : (L, N) complex double — column-per-sample
%                     (transposed inside this function before write)
%   labels          : (N, 1) integer in 1..num_classes (1-based MATLAB
%                     convention; converted to 0-based on write)
%   snr_db          : (N, 1) double
%   pulse_widths_us : (N, 1) double
%   cfg             : struct from generation_config()
%
% Notes
% -----
% - HDF5 doesn't have a native complex type. We split the signal into
%   real/imag parts and store as a (N, L, 2) float32 dataset, OR use the
%   convention recommended for h5py: a compound type with 'r' and 'i'
%   fields. Here we use the (N, L, 2) approach since it is widely
%   compatible with both MATLAB (h5create/h5write) and Python (h5py).
% - Python side: load with
%       import h5py, numpy as np
%       with h5py.File('dataset.h5','r') as f:
%           sig = f['signals'][:]          # (N, L, 2) float32
%           sig = sig[...,0] + 1j*sig[...,1]   # to complex
% - Labels are stored 0-based to match Python conventions; class_names
%   provides the lookup.

    if nargin < 6
        error('save_dataset_h5:missingArgs', ...
              'cfg must be provided as the 6th argument');
    end

    % Make sure output directory exists
    out_dir = fileparts(filepath);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Delete existing file if any (h5create cannot overwrite)
    if exist(filepath, 'file')
        delete(filepath);
    end

    % --- Dimensions -----------------------------------------------------
    [L, N] = size(signals);              % L=signal length, N=#samples
    if L ~= cfg.N
        warning('save_dataset_h5:unexpectedLength', ...
                'Signal length (%d) does not match cfg.N (%d).', L, cfg.N);
    end

    % --- Convert complex -> (N, L, 2) float32 ---------------------------
    % MATLAB stores arrays column-major; we want one sample per row.
    % After permutation: (N, L) complex -> (N, L, 2) float32 with
    % the last dim being [real, imag].
    signals_t = signals.';                              % (N, L) complex
    sig_real  = single(real(signals_t));
    sig_imag  = single(imag(signals_t));
    sig_h5    = cat(3, sig_real, sig_imag);             % (N, L, 2) float32

    % --- Convert labels: 1-based -> 0-based, uint8 ----------------------
    labels_h5 = uint8(labels(:) - 1);

    % --- SNR and pulse widths as float32 --------------------------------
    snr_h5 = single(snr_db(:));
    pw_h5  = single(pulse_widths_us(:));

    % --- Write datasets -------------------------------------------------
    % h5create signature: h5create(file, dataset, dims, 'Datatype', type)
    % MATLAB writes in column-major; we pass dims matching our arrays.

    h5create(filepath, '/signals', size(sig_h5), 'Datatype', 'single', ...
             'Deflate', 4, 'ChunkSize', [min(N, 256), L, 2]);
    h5write(filepath, '/signals', sig_h5);

    h5create(filepath, '/labels', size(labels_h5), 'Datatype', 'uint8');
    h5write(filepath, '/labels', labels_h5);

    h5create(filepath, '/snr_db', size(snr_h5), 'Datatype', 'single');
    h5write(filepath, '/snr_db', snr_h5);

    h5create(filepath, '/pulse_widths_us', size(pw_h5), 'Datatype', 'single');
    h5write(filepath, '/pulse_widths_us', pw_h5);

    % --- Class names as variable-length string dataset ------------------
    % Use H5T low-level API for portable string handling.
    file_id = H5F.open(filepath, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    type_id = H5T.copy('H5T_C_S1');
    H5T.set_size(type_id, 'H5T_VARIABLE');
    space_id = H5S.create_simple(1, numel(cfg.class_names), ...
                                 numel(cfg.class_names));
    dset_id = H5D.create(file_id, 'class_names', type_id, space_id, ...
                         'H5P_DEFAULT');
    H5D.write(dset_id, type_id, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', ...
              cfg.class_names);
    H5D.close(dset_id);
    H5S.close(space_id);
    H5T.close(type_id);
    H5F.close(file_id);

    % --- Root attributes (metadata) -------------------------------------
    h5writeatt(filepath, '/', 'sample_rate_hz',    cfg.fs);
    h5writeatt(filepath, '/', 'signal_length',     int32(cfg.N));
    h5writeatt(filepath, '/', 'master_seed',       int32(cfg.master_seed));
    h5writeatt(filepath, '/', 'generation_date',   ...
               datestr(now, 'yyyy-mm-ddTHH:MM:SS'));
    h5writeatt(filepath, '/', 'dataset_version',   cfg.dataset_version);
    h5writeatt(filepath, '/', 'samples_per_class', ...
               int32(cfg.samples_per_class));
    h5writeatt(filepath, '/', 'num_classes',       int32(cfg.num_classes));
    h5writeatt(filepath, '/', 'snr_db_min',        cfg.snr_db_min);
    h5writeatt(filepath, '/', 'snr_db_max',        cfg.snr_db_max);
    h5writeatt(filepath, '/', 'snr_db_step',       cfg.snr_db_step);
    h5writeatt(filepath, '/', 'pulse_width_us_min', cfg.pulse_width_us(1));
    h5writeatt(filepath, '/', 'pulse_width_us_max', cfg.pulse_width_us(2));
    h5writeatt(filepath, '/', 'storage_convention', ...
               'signals stored as (N, L, 2) float32; last dim = [real, imag]');

    if cfg.verbose
        info = dir(filepath);
        fprintf('Saved dataset to %s (%.1f MB)\n', filepath, ...
                info.bytes / 1024 / 1024);
    end
end
