package anthropic

import (
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
)

const advisorToolJSON = `{"type":"advisor_20260301","name":"advisor","model":"claude-opus-4-8","max_uses":3,"max_tokens":2048,"caching":{"type":"ephemeral","ttl":"5m"}}`

// TestAnthropicTool_AdvisorMarshalRoundTrip verifies the advisor tool survives
// Unmarshal/Marshal through AnthropicTool — in particular that the unique
// `model` field is preserved despite `max_uses` being shared (embedded) with
// the web_search/web_fetch variant structs.
func TestAnthropicTool_AdvisorMarshalRoundTrip(t *testing.T) {
	var tool AnthropicTool
	if err := sonic.Unmarshal([]byte(advisorToolJSON), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tool.Type == nil || *tool.Type != AnthropicToolTypeAdvisor20260301 {
		t.Fatalf("type not parsed: %+v", tool.Type)
	}
	if tool.AnthropicToolAdvisor == nil {
		t.Fatal("AnthropicToolAdvisor is nil after unmarshal")
	}
	if tool.AnthropicToolAdvisor.Model != "claude-opus-4-8" {
		t.Errorf("model = %q, want claude-opus-4-8", tool.AnthropicToolAdvisor.Model)
	}
	if tool.AnthropicToolAdvisor.MaxTokens == nil || *tool.AnthropicToolAdvisor.MaxTokens != 2048 {
		t.Errorf("max_tokens not parsed: %+v", tool.AnthropicToolAdvisor.MaxTokens)
	}
	if tool.AnthropicToolAdvisor.Caching == nil || tool.AnthropicToolAdvisor.Caching.TTL != "5m" {
		t.Errorf("caching not parsed: %+v", tool.AnthropicToolAdvisor.Caching)
	}

	out, err := sonic.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(out)
	for _, want := range []string{`"type":"advisor_20260301"`, `"model":"claude-opus-4-8"`, `"max_tokens":2048`, `"ttl":"5m"`} {
		if !strings.Contains(s, want) {
			t.Errorf("marshaled JSON missing %q: %s", want, s)
		}
	}
}

// TestAdvisor_ResponsesRoundTrip verifies the advisor tool round-trips through
// the neutral ResponsesTool schema (Anthropic -> Bifrost -> Anthropic) with its
// type, name, and model intact.
func TestAdvisor_ResponsesRoundTrip(t *testing.T) {
	var tool AnthropicTool
	if err := sonic.Unmarshal([]byte(advisorToolJSON), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	bifrostTool := convertAnthropicToolToBifrost(&tool)
	if bifrostTool == nil {
		t.Fatal("convertAnthropicToolToBifrost returned nil")
	}
	if bifrostTool.Type != schemas.ResponsesToolType(AnthropicToolTypeAdvisor20260301) {
		t.Fatalf("neutral type = %q, want advisor_20260301", bifrostTool.Type)
	}
	if bifrostTool.ResponsesToolAdvisor == nil || bifrostTool.ResponsesToolAdvisor.Model != "claude-opus-4-8" {
		t.Fatalf("neutral advisor model not carried: %+v", bifrostTool.ResponsesToolAdvisor)
	}

	back := convertBifrostToolToAnthropic("claude-sonnet-4-6", bifrostTool, schemas.Anthropic, false)
	if back == nil || back.Type == nil || *back.Type != AnthropicToolTypeAdvisor20260301 {
		t.Fatalf("rebuilt type wrong: %+v", back)
	}
	if back.Name != string(AnthropicToolNameAdvisor) {
		t.Errorf("rebuilt name = %q, want advisor", back.Name)
	}
	if back.AnthropicToolAdvisor == nil || back.AnthropicToolAdvisor.Model != "claude-opus-4-8" {
		t.Fatalf("rebuilt advisor model lost: %+v", back.AnthropicToolAdvisor)
	}
}

// TestAdvisor_ProviderGating verifies advisor is supported only on Anthropic.
func TestAdvisor_ProviderGating(t *testing.T) {
	cases := []struct {
		provider schemas.ModelProvider
		want     bool
	}{
		{schemas.Anthropic, true},
		{schemas.Vertex, false},
		{schemas.Bedrock, false},
		{schemas.Azure, false},
	}
	for _, c := range cases {
		got := isAnthropicServerToolSupported(string(AnthropicToolTypeAdvisor20260301), ProviderFeatures[c.provider])
		if got != c.want {
			t.Errorf("isAnthropicServerToolSupported advisor on %s = %v, want %v", c.provider, got, c.want)
		}

		err := ValidateToolsForProvider([]schemas.ResponsesTool{{
			Type: schemas.ResponsesToolType(AnthropicToolTypeAdvisor20260301),
		}}, c.provider)
		if c.want && err != nil {
			t.Errorf("ValidateToolsForProvider advisor on %s returned error: %v", c.provider, err)
		}
		if !c.want && err == nil {
			t.Errorf("ValidateToolsForProvider advisor on %s should have errored", c.provider)
		}
	}
}

// TestAdvisor_BetaHeaderFiltering verifies the advisor beta header survives for
// Anthropic and is dropped for providers that don't support advisor.
func TestAdvisor_BetaHeaderFiltering(t *testing.T) {
	in := []string{AnthropicAdvisorBetaHeader}

	if got := FilterBetaHeadersForProvider(in, schemas.Anthropic); len(got) != 1 || got[0] != AnthropicAdvisorBetaHeader {
		t.Errorf("Anthropic should keep advisor header, got %v", got)
	}
	for _, p := range []schemas.ModelProvider{schemas.Vertex, schemas.Bedrock, schemas.Azure} {
		if got := FilterBetaHeadersForProvider(in, p); len(got) != 0 {
			t.Errorf("%s should drop advisor header, got %v", p, got)
		}
	}
}
