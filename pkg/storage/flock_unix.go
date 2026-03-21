//go:build !windows

package storage

import (
	"os"
	"syscall"
)

// flockExclusive pose un verrou exclusif POSIX sur le fichier.
// Bloque jusqu'à ce que le verrou soit disponible (LOCK_EX sans LOCK_NB).
// Utilisé pour sérialiser les écritures dans vectors.bin entre process.
func flockExclusive(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
}

// flockUnlock libère le verrou POSIX sur le fichier.
func flockUnlock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
