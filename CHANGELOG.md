# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.16...HEAD)</small>

<!-- insertion marker -->
## [v0.1.16](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.16) - 2026-11-18

<small>[Compare with v0.1.15](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.15...v0.1.16)</small>

## [v0.1.15](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.15) - 2026-07-28

<small>[Compare with v0.1.14](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.14...v0.1.15)</small>

### Fixed

- Fix opto stimulus timing ([a02bd00](https://github.com/AllenInstitute/npc_samstim/commit/a02bd0091ae0a8e2b8087a2eb74f285947f97684) by Codex).

## [v0.1.14](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.14) - 2026-06-12

<small>[Compare with v0.1.13](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.13...v0.1.14)</small>

## [v0.1.13](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.13) - 2025-07-28

<small>[Compare with v0.1.12](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.12...v0.1.13)</small>

## [v0.1.12](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.12) - 2025-07-28

<small>[Compare with v0.1.11](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.11...v0.1.12)</small>

## [v0.1.11](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.11) - 2025-03-07

<small>[Compare with v0.1.10](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.10...v0.1.11)</small>

### Fixed

- Fix missing opto line error handling ([62aadfb](https://github.com/AllenInstitute/npc_samstim/commit/62aadfb83f3b3d3d237c9dd94f67be509581bf35) by bjhardcastle).

## [v0.1.10](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.10) - 2024-12-04

<small>[Compare with v0.1.9](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.9...v0.1.10)</small>

### Fixed

- Fix opto alignment bug - trials being skipped due to onset frame == 0 (introduced in 0.1.9) ([871b093](https://github.com/AllenInstitute/npc_samstim/commit/871b0934689ba63c07efb5acf6a2bdf0e82f1588) by bjhardcastle).

## [v0.1.9](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.9) - 2024-10-23

<small>[Compare with v0.1.8](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.8...v0.1.9)</small>

### Fixed

- Fix getting opto waveforms ([be44444](https://github.com/AllenInstitute/npc_samstim/commit/be44444f9a380098883e7110e4048d35b8d4fa9b) by bjhardcastle).

## [v0.1.8](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.8) - 2024-10-22

<small>[Compare with v0.1.7](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.7...v0.1.8)</small>

### Fixed

- Fix getting waveform timestamps ([b6d04c8](https://github.com/AllenInstitute/npc_samstim/commit/b6d04c8491ed3b0d59472ff2528e497b69246cda) by bjhardcastle).

## [v0.1.7](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.7) - 2024-10-03

<small>[Compare with v0.1.6](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.6...v0.1.7)</small>

### Fixed

- Fix progress bar on threaded waveform fetching ([c78f1c1](https://github.com/AllenInstitute/npc_samstim/commit/c78f1c19487417f6bea07dcd4a09b84bba722b48) by bjhardcastle).

## [v0.1.6](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.6) - 2024-10-03

<small>[Compare with v0.1.5](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.5...v0.1.6)</small>

### Fixed

- Fix converting iterable to array ([61273bd](https://github.com/AllenInstitute/npc_samstim/commit/61273bd3984533770a1c2920f0da09c44e40a5d6) by bjhardcastle).

## [v0.1.5](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.5) - 2024-10-03

<small>[Compare with v0.1.4](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.4...v0.1.5)</small>

### Added

- Add `get_waveforms_from_nidaq_recording` - for capturing audio or opto traces ([d4614e8](https://github.com/AllenInstitute/npc_samstim/commit/d4614e8e6dc49fad181e0b255858f7ced7c524fb) by bjhardcastle).
- Add doctest runner in module ([a40e9e6](https://github.com/AllenInstitute/npc_samstim/commit/a40e9e6dce4e2f663fe04b097a3eb9797a546fd1) by bjhardcastle).

### Fixed

- Fix test ([c0ee1df](https://github.com/AllenInstitute/npc_samstim/commit/c0ee1dfe9ee3c5ae536e8335efc26af1f3a50471) by bjhardcastle).
- Fix docstring ([697cd88](https://github.com/AllenInstitute/npc_samstim/commit/697cd88f5351fd8db7bf8faf101e4c61eae01e33) by bjhardcastle).

## [v0.1.4](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.4) - 2024-03-30

<small>[Compare with v0.1.3](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.3...v0.1.4)</small>

### Fixed

- Fix previous commit ([df86c95](https://github.com/AllenInstitute/npc_samstim/commit/df86c9525d66f40a454e2d63645cb41778e60564) by bjhardcastle).

## [v0.1.3](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.3) - 2024-03-30

<small>[Compare with v0.1.2](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.2...v0.1.3)</small>

## [v0.1.2](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.2) - 2024-02-02

<small>[Compare with v0.1.1](https://github.com/AllenInstitute/npc_samstim/compare/v0.1.1...v0.1.2)</small>

### Fixed

- Fix type ([325700f](https://github.com/AllenInstitute/npc_samstim/commit/325700f29d352fb4448d6011f94cec54ddf2b03c) by bjhardcastle).

## [v0.1.1](https://github.com/AllenInstitute/npc_samstim/releases/tag/v0.1.1) - 2024-02-02

<small>[Compare with first commit](https://github.com/AllenInstitute/npc_samstim/compare/3b3612bb73efc3fa51fc8aaa03dbaf81510eb3a0...v0.1.1)</small>
