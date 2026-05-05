% TEST_STEPPED_FH_VISUALIZATION
% Sanity check for stepped frequency-hopping generator. Produces:
%   - Figure 1: 6 stepped pulses (mix of N=5/6/7/8, up/down).
%               Inst. freq should be N monotonic plateaus; spectrogram
%               should be a stair-step pattern.
%   - Figure 2: Costas vs Stepped vs LFM side-by-side comparison.
%               This is the discriminative challenge for the model.
%   - Figure 3: 8-class comparison — full lineup of all Module A signals.
%               (The "Module A complete" overview figure.)
%   - Figure 4: Stepped at multiple SNRs.
%
% Run from data_generation/matlab/ directory:
%   >> cd data_generation/matlab
%   >> addpath(genpath(pwd))
%   >> test_stepped_fh_visualization

clear; clc; close all;

addpath('config', 'signals', 'utils');

cfg = generation_config();

% ------------------------------------------------------------------
% Figure 1: 6 stepped pulses (clean)
% ------------------------------------------------------------------
figure('Name', 'Stepped/FH clean pulses', 'Position', [60 60 1300 800]);

N_count = struct('N5', 0, 'N6', 0, 'N7', 0, 'N8', 0);
dir_count = struct('up', 0, 'down', 0);

for k = 1:6
    [s, p] = generate_stepped_fh(cfg);
    N_count.(sprintf('N%d', p.N)) = N_count.(sprintf('N%d', p.N)) + 1;
    dir_count.(p.direction) = dir_count.(p.direction) + 1;

    t = (0:cfg.N-1) * cfg.Ts * 1e6;     % us

    subplot(6, 3, (k-1)*3 + 1);
    plot(t, real(s), 'b'); hold on; plot(t, imag(s), 'r');
    xlabel('t [\mus]'); ylabel('amp');
    title(sprintf('Stepped #%d  N=%d  [%s]  T=%.2f \\mus  \\Delta f=%.1f MHz', ...
          k, p.N, p.direction, p.pulse_width_s*1e6, p.delta_f_hz/1e6));
    legend('I','Q', 'Location', 'eastoutside'); grid on;

    % Instantaneous frequency — should be N monotonic plateaus
    subplot(6, 3, (k-1)*3 + 2);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1);
    active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot(t, inst_freq/1e6, 'w', 'LineWidth', 1.4); hold on;
    for f_hop = p.frequencies_hz
        yline(f_hop/1e6, 'Color', [0.4 0.9 0.4], ...
              'LineStyle', '--', 'Alpha', 0.7);
    end
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title(sprintf('Inst. freq (N=%d steps, %s)', p.N, p.direction));
    grid on;

    subplot(6, 3, (k-1)*3 + 3);
    spectrogram(s, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title('STFT spectrogram');
end

fprintf('Mix in Figure 1: N counts = (%d,%d,%d,%d) for N=5/6/7/8; up=%d, down=%d\n', ...
        N_count.N5, N_count.N6, N_count.N7, N_count.N8, ...
        dir_count.up, dir_count.down);

% ------------------------------------------------------------------
% Figure 2: Costas vs Stepped vs LFM — the discriminative challenge
% ------------------------------------------------------------------
figure('Name', 'Costas vs Stepped vs LFM', 'Position', [120 30 1300 700]);

[s_lfm, p_lfm] = generate_lfm(cfg);
[s_cos, p_cos] = generate_costas(cfg);
[s_stp, p_stp] = generate_stepped_fh(cfg);

trio = {
    s_lfm, p_lfm, sprintf('LFM (%s, B=%.1f MHz)', p_lfm.direction, ...
                          p_lfm.bandwidth_hz/1e6),    'continuous';
    s_stp, p_stp, sprintf('Stepped (%s, N=%d, \\Delta f=%.1f MHz)', ...
                          p_stp.direction, p_stp.N, p_stp.delta_f_hz/1e6), ...
                                                       'monotonic-discrete';
    s_cos, p_cos, sprintf('Costas (N=%d, \\Delta f=%.1f MHz)', ...
                          p_cos.N, p_cos.delta_f_hz/1e6), ...
                                                       'permuted-discrete'
};

for k = 1:3
    s = trio{k,1}; p = trio{k,2}; name = trio{k,3}; kind = trio{k,4};

    subplot(3, 2, (k-1)*2 + 1);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot((0:cfg.N-1)*cfg.Ts*1e6, inst_freq/1e6, 'w', 'LineWidth', 1.4);
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title(sprintf('%s  -  Inst. freq  [%s]', name, kind));
    grid on;

    subplot(3, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 3: 8-class comparison (Module A complete!)
% ------------------------------------------------------------------
figure('Name', '8-class comparison (Module A complete)', ...
       'Position', [180 5 1300 1000]);

[s_lfm, p_lfm] = generate_lfm(cfg);

s_nlfm = []; p_nlfm = [];
for trial = 1:50
    [s_, p_] = generate_nlfm(cfg);
    if strcmp(p_.variant, 'quadratic'), s_nlfm=s_; p_nlfm=p_; break; end
end

[s_brk, p_brk] = generate_barker(cfg);
[s_frk, p_frk] = generate_frank(cfg);
[s_pp,  p_pp]  = generate_polyphase(cfg);
[s_cos, p_cos] = generate_costas(cfg);
[s_cw,  p_cw]  = generate_cw(cfg);
[s_stp, p_stp] = generate_stepped_fh(cfg);

octet = {
    s_lfm,  p_lfm,  'LFM (linear)';
    s_nlfm, p_nlfm, 'NLFM (quadratic)';
    s_brk,  p_brk,  sprintf('Barker (%s)', p_brk.code_name);
    s_frk,  p_frk,  sprintf('Frank (N=%d)', p_frk.N);
    s_pp,   p_pp,   sprintf('Polyphase (%s, N=%d)', p_pp.subcode, p_pp.N);
    s_cos,  p_cos,  sprintf('Costas (N=%d)', p_cos.N);
    s_cw,   p_cw,   sprintf('CW (f_c=%.1f MHz)', p_cw.f_carrier_hz/1e6);
    s_stp,  p_stp,  sprintf('Stepped (%s, N=%d)', p_stp.direction, p_stp.N)
};

for k = 1:8
    s = octet{k,1}; p = octet{k,2}; name = octet{k,3};

    subplot(8, 2, (k-1)*2 + 1);
    inst_phase = unwrap(angle(s));
    inst_freq  = [0; diff(inst_phase)] / (2*pi*cfg.Ts);
    active = false(cfg.N,1); active(p.start_idx:p.stop_idx) = true;
    inst_freq(~active) = NaN;
    plot((0:cfg.N-1)*cfg.Ts*1e6, inst_freq/1e6, 'w', 'LineWidth', 1.0);
    xlabel('t [\mus]'); ylabel('f_{inst} [MHz]');
    title([name, '  -  Instantaneous frequency']);
    grid on;

    subplot(8, 2, (k-1)*2 + 2);
    spectrogram(s, hamming(128), 120, 256, cfg.fs, 'yaxis', 'centered');
    title([name, '  -  STFT spectrogram']);
end

% ------------------------------------------------------------------
% Figure 4: Stepped at multiple SNRs
% ------------------------------------------------------------------
figure('Name', 'Stepped/FH at different SNRs', 'Position', [240 60 1200 700]);
snr_test = [-10, 0, 10, 20];

for i = 1:numel(snr_test)
    s_noisy = add_awgn(s_stp, snr_test(i), [p_stp.start_idx, p_stp.stop_idx]);

    subplot(numel(snr_test), 2, 2*i - 1);
    t = (0:cfg.N-1) * cfg.Ts * 1e6;
    plot(t, real(s_noisy));
    title(sprintf('Stepped (N=%d, %s), real part, SNR = %d dB', ...
          p_stp.N, p_stp.direction, snr_test(i)));
    xlabel('t [\mus]'); ylabel('amp'); grid on;

    subplot(numel(snr_test), 2, 2*i);
    spectrogram(s_noisy, hamming(64), 56, 256, cfg.fs, 'yaxis', 'centered');
    title(sprintf('Stepped spectrogram, SNR = %d dB', snr_test(i)));
end

fprintf('Tests complete. Inspect four figures.\n');
fprintf('  - Figure 1: inst. freq must be N monotonic plateaus.\n');
fprintf('  - Figure 2: LFM=smooth, Stepped=monotonic stair, Costas=permuted.\n');
fprintf('  - Figure 3: 8-class lineup; Module A complete!\n');
fprintf('  - Figure 4: stair pattern fades into noise at SNR = -10 dB.\n');
