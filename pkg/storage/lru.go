package storage

import "container/list"

// lruCache est un cache LRU thread-unsafe (protégé par le mutex du Store).
// Clé : chunk ID (string), Valeur : vecteur float32.
// Quand la capacité est dépassée, l'entrée la moins récemment utilisée est évincée.
// Capacité ≤ 0 signifie illimitée (comportement identique à l'ancienne map).
type lruCache struct {
	cap   int
	ll    *list.List
	items map[string]*list.Element
}

type lruEntry struct {
	key string
	vec []float32
}

func newLRUCache(cap int) *lruCache {
	return &lruCache{
		cap:   cap,
		ll:    list.New(),
		items: make(map[string]*list.Element),
	}
}

// get retourne le vecteur associé à la clé et le déplace en tête (MRU).
func (c *lruCache) get(key string) ([]float32, bool) {
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.ll.MoveToFront(el)
	return el.Value.(*lruEntry).vec, true
}

// set insère ou met à jour une entrée. Évince le LRU si nécessaire.
func (c *lruCache) set(key string, vec []float32) {
	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		el.Value.(*lruEntry).vec = vec
		return
	}
	el := c.ll.PushFront(&lruEntry{key: key, vec: vec})
	c.items[key] = el

	if c.cap > 0 && c.ll.Len() > c.cap {
		c.evict()
	}
}

// evict supprime l'entrée la moins récemment utilisée (queue de la liste).
func (c *lruCache) evict() {
	tail := c.ll.Back()
	if tail == nil {
		return
	}
	c.ll.Remove(tail)
	delete(c.items, tail.Value.(*lruEntry).key)
}

// len retourne le nombre d'entrées actuellement en cache.
func (c *lruCache) len() int {
	return c.ll.Len()
}

// keys retourne toutes les clés présentes en cache.
// Utilisé pour la reconstruction de l'index HNSW au démarrage.
func (c *lruCache) keys() []string {
	keys := make([]string, 0, c.ll.Len())
	for k := range c.items {
		keys = append(keys, k)
	}
	return keys
}

// forEach itère sur toutes les entrées (ordre non garanti).
// Utilisé pour la construction de l'index HNSW.
func (c *lruCache) forEach(fn func(key string, vec []float32)) {
	for el := c.ll.Front(); el != nil; el = el.Next() {
		e := el.Value.(*lruEntry)
		fn(e.key, e.vec)
	}
}
