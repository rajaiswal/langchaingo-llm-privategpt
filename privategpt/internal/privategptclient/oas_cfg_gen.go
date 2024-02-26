// Code generated by ogen, DO NOT EDIT.

package privategptclient

import (
	"net/http"

	ht "github.com/ogen-go/ogen/http"
)

type (
	optionFunc[C any] func(*C)
)

type clientConfig struct {
	Client ht.Client
}

// ClientOption is client config option.
type ClientOption interface {
	applyClient(*clientConfig)
}

var _ ClientOption = (optionFunc[clientConfig])(nil)

func (o optionFunc[C]) applyClient(c *C) {
	o(c)
}

func newClientConfig(opts ...ClientOption) clientConfig {
	cfg := clientConfig{
		Client: http.DefaultClient,
	}
	for _, opt := range opts {
		opt.applyClient(&cfg)
	}
	return cfg
}

type baseClient struct {
	cfg clientConfig
}

func (cfg clientConfig) baseClient() (c baseClient, err error) {
	c = baseClient{cfg: cfg}
	return c, nil
}

// Option is config option.
type Option interface {
	ClientOption
}

// WithClient specifies http client to use.
func WithClient(client ht.Client) ClientOption {
	return optionFunc[clientConfig](func(cfg *clientConfig) {
		if client != nil {
			cfg.Client = client
		}
	})
}
