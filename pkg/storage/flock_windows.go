//go:build windows

package storage

import "os"

// flock non supporté sur Windows — no-op.
// vectors.bin est un fichier local mono-process sur cette plateforme.
func flockExclusive(f *os.File) error { return nil }
func flockUnlock(f *os.File) error    { return nil }
