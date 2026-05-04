function s_norm = normalize_signal(s)
% NORMALIZE_SIGNAL  Scale a signal to unit average power.
%
% s_norm = s / sqrt(mean(|s|^2))
%
% Used inside per-class generators if cfg.normalize_signal_power is true,
% but exposed as a utility for tests and any post-hoc rescaling.
%
% If the input has zero power, it is returned unchanged.

    s = s(:);
    p = mean(abs(s).^2);
    if p > 0
        s_norm = s / sqrt(p);
    else
        s_norm = s;
    end
end
