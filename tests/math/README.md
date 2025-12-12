# Math Test Suite Notes

This directory contains Vitest suites that validate the numerical helpers used throughout the project.

## GPU parity

Softmax parity tests will run when `navigator.gpu` is available. In environments without WebGPU the tests are skipped; CPU reference values remain covered, but GPU parity cannot be validated in that setting.
