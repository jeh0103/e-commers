function jsonPost(url, payload) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "요청을 처리하지 못했습니다.");
    }
    return data;
  });
}

function formPayload(form) {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = value;
  }
  return payload;
}

document.addEventListener("submit", async (event) => {
  const actionForm = event.target.closest("[data-action-form]");
  if (actionForm) {
    event.preventDefault();
    const customerId = actionForm.dataset.customerId;
    const toast = document.querySelector("[data-action-toast]");
    try {
      await jsonPost(`/api/customers/${encodeURIComponent(customerId)}/actions`, formPayload(actionForm));
      if (toast) {
        toast.textContent = "조치 기록이 저장되었습니다. 새로고침하면 처리 이력에서 확인할 수 있습니다.";
        toast.className = "toast";
        toast.style.display = "block";
      }
      actionForm.reset();
    } catch (error) {
      if (toast) {
        toast.textContent = error.message;
        toast.className = "toast error";
        toast.style.display = "block";
      }
    }
  }

  const smsForm = event.target.closest("[data-sms-form]");
  if (smsForm) {
    event.preventDefault();
    const customerId = smsForm.dataset.customerId;
    const output = document.querySelector("[data-sms-output]");
    const meta = document.querySelector("[data-sms-meta]");
    try {
      const payload = formPayload(smsForm);
      payload.target_segments = Number(payload.target_segments || 1);
      const data = await jsonPost(`/api/customers/${encodeURIComponent(customerId)}/sms-preview`, payload);
      if (output) output.value = data.message || "";
      if (meta && data.segments) {
        meta.textContent = `${data.segments.length}자 · 추정 ${data.segments.segments}건 · 남은 ${data.segments.remaining}자`;
      }
    } catch (error) {
      if (meta) meta.textContent = error.message;
    }
  }
});

document.addEventListener("click", async (event) => {
  const copyButton = event.target.closest("[data-copy-sms]");
  if (!copyButton) return;
  const output = document.querySelector("[data-sms-output]");
  if (!output) return;
  await navigator.clipboard.writeText(output.value);
  copyButton.textContent = "복사됨";
  window.setTimeout(() => {
    copyButton.textContent = "문자 복사";
  }, 1400);
});

document.addEventListener("submit", (event) => {
  const search = event.target.closest("[data-customer-search]");
  if (!search) return;
  event.preventDefault();
  const input = search.querySelector("input[name='customer_id']");
  const value = input ? input.value.trim() : "";
  if (value) {
    window.location.href = `/customers/${encodeURIComponent(value)}`;
  }
});
