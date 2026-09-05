function splitMarkdownRow(lined) {
    let body = String(lined ?? '').trim();
    if (body.startsWith('|')) {
      body = body.slice(1);
    }
    if (body.endsWith('|')) {
      body = body.slice(0, -1);
    }
    const cells = body.split('|').map((cell) => cell.trim());
    while (cells.length > 0 && cells[0] === '') cells.shift();
    while (cells.length > 0 && cells[cells.length - 1] === '') cells.pop();
    return cells;
  }
  function isTableRow(line) {
    const t = String(line ?? '').trim();
    if (!t) return false;
    if (t.startsWith('|')) return true;
    return t.endsWith('|') && t.split('|').length >= 3;
  }
  function isTableSeparator(line) {
    const cells = splitMarkdownRow(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{2,}:?$/.test(cell));
  }

document.addEventListener('DOMContentLoaded', () => {
  const BASE =
    window.RAG_API_BASE || 'http://127.0.0.1:8000';
  const $ = (selector) => document.querySelector(selector);
  const chat = $('#chat');
  const input = $('#user-input');
  const sendBtn = $('#send-btn');
  const chatForm = $('#chat-form');
  const welcome = $('#welcome');
  const chatTitle = $('#chat-title');
  const suggestions = $('#suggestions');
  const newChatBtn = $('#new-chat');
  const clearChatBtn = $('#clear-chat');
  const fileInput = $('#file-input');
  const documentList = $('#document-list');
  const documentCount = $('#document-count');
  const uploadStatus = $('#upload-status');
  const conversationList = $('#conversation-list');
  let busy = false;
  function createConversationId() {
    if (
      typeof crypto !== 'undefined' &&
      typeof crypto.randomUUID === 'function'
    ) {
      return crypto.randomUUID();
    }
    return `chat-${Date.now()}-${Math.random()
      .toString(36)
      .substring(2, 9)}`;
  }
  let conversationId = createConversationId();
  function safe(value) {
    const element = document.createElement('span');
    element.textContent = value ?? '';
    return element.innerHTML;
  }
  function apiUrl(path) {
    return `${BASE}${path}`;
  }
  function renderInline(element, text) {
    if (!text) return;
    const regex = /(\*\*.+?\*\*|\*\(.+?\)\*)/g;
    let last = 0;
    let match;
    while ((match = regex.exec(text)) !== null) {
      if (match.index > last) {
        element.appendChild(
          document.createTextNode(text.slice(last, match.index))
        );
      }
      const token = match[0];
      if (token.startsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = token.slice(2, -2);
        element.appendChild(strong);
      } else {
        const cite = document.createElement('span');
        cite.className = 'source-ref';
        cite.textContent = token.slice(1, -1);
        element.appendChild(cite);
      }
      last = match.index + token.length;
    }
    if (last < text.length) {
      element.appendChild(document.createTextNode(text.slice(last)));
    }
  }
  function renderMarkdown(container, text) {
    container.textContent = '';
    const lines = String(text ?? '').split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i].trim();
      if (!line) {
        i++;
        continue;
      }
      if (isTableRow(line)) {
        const table = document.createElement('table');
        table.className = 'answer-table';
        let headerDone = false;
        let headerCells = null;
        while (i < lines.length && isTableRow(lines[i])) {
          const rowLine = lines[i];
          if (isTableSeparator(rowLine)) {
            i++;
            continue;
          }
          const cells = splitMarkdownRow(rowLine);
          if (headerCells === null) headerCells = cells;
          while (cells.length < headerCells.length) cells.push('');
          cells.length = headerCells.length;
          const row = document.createElement('tr');
          cells.forEach((cell, cellIndex) => {
            const cellNode = document.createElement(
              !headerDone && cellIndex === 0 ? 'th' : 'td'
            );
            renderInline(cellNode, cell);
            row.appendChild(cellNode);
          });
          table.appendChild(row);
          headerDone = true;
          i++;
        }
        container.appendChild(table);
        continue;
      }
      if (/^[•\-]\s/.test(line)) {
        const list = document.createElement('ul');
        while (i < lines.length && /^[•\-]\s/.test(lines[i].trim())) {
          const item = document.createElement('li');
          renderInline(
            item,
            lines[i].trim().replace(/^[•\-]\s/, '')
          );
          list.appendChild(item);
          i++;
        }
        container.appendChild(list);
        continue;
      }
      const paragraph = document.createElement('p');
      renderInline(paragraph, line);
      container.appendChild(paragraph);
      i++;
    }
  }
  function resizeInput() {
    if (!input) return;
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  }
  function scrollChatToBottom() {
    if (!chat) return;
    chat.scrollTop = chat.scrollHeight;
  }
  function updateSendButton() {
    if (!sendBtn) return;
    sendBtn.disabled = busy || !input.value.trim();
  }
  function message(text, role = 'assistant', sources = []) {
    if (!chat) return null;
    if (welcome) {
      welcome.hidden = true;
    }
    const article = document.createElement('article');
    article.className = `message ${role}`;
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'You' : '✦';
    const body = document.createElement('div');
    body.className = 'message-body';
    renderMarkdown(body, text ?? '');
    if (Array.isArray(sources) && sources.length > 0) {
      const sourcesContainer = document.createElement('div');
      sourcesContainer.className = 'sources';
      const seenSources = new Set();
      sources.forEach((source) => {
        const sourceName = source?.source ?? 'Unknown source';
        if (!source?.source) return;
        const page = source?.page;
        const dedupeKey = `${sourceName}|${page ?? ''}`;
        if (seenSources.has(dedupeKey)) return;
        seenSources.add(dedupeKey);
        const sourceElement = document.createElement('span');
        sourceElement.className = 'source';
        const parts = [sourceName];
        if (Number.isFinite(page)) {
          parts.push(`p. ${page}`);
        }
        sourceElement.textContent = parts.join(' · ');
        sourcesContainer.appendChild(sourceElement);
      });
      if (seenSources.size > 0) {
        body.appendChild(sourcesContainer);
      }
    }
    article.appendChild(avatar);
    article.appendChild(body);
    chat.appendChild(article);
    scrollChatToBottom();
    return article;
  }
  function thinking() {
    if (!chat) return null;
    const article = document.createElement('article');
    article.className = 'message thinking-message';
    const body = document.createElement('div');
    body.className = 'message-body';
    const label = document.createElement('div');
    label.className = 'thinking-label';
    label.textContent = 'Generating answer…';
    const dots = document.createElement('div');
    dots.className = 'thinking';
    dots.innerHTML = '<i></i><i></i><i></i>';
    body.appendChild(label);
    body.appendChild(dots);
    article.appendChild(body);
    chat.appendChild(article);
    scrollChatToBottom();
    return article;
  }
  async function send(text) {
    if (!text || busy) return;
    message(text, 'user');
    input.value = '';
    resizeInput();
    updateSendButton();
    busy = true;
    input.disabled = true;
    if (sendBtn) {
      sendBtn.disabled = true;
    }
    const wait = thinking();
    try {
      const response = await fetch(apiUrl('/api/chat'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId,
          language: 'en',
          model: 'gemma'
        })
      });
      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }
      if (wait) {
        wait.remove();
      }
      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          `Request failed (${response.status})`
        );
      }
      const answer =
        data.answer ||
        data.response ||
        'No response received.';
      const sources =
        Array.isArray(data.sources)
          ? data.sources
          : [];
      message(answer, 'assistant', sources);
      if (chatTitle) {
        const title =
          text.length > 34
            ? `${text.slice(0, 31)}…`
            : text;
        chatTitle.textContent = title;
      }
      loadHistory();
    } catch (error) {
      if (wait) {
        wait.remove();
      }
      console.error('Chat API error:', error);
      if (error instanceof TypeError) {
        message(
          `Could not reach the API at ${BASE}. ` +
          `Make sure your FastAPI backend is running.`,
          'error'
        );
      } else {
        message(
          error.message || 'Something went wrong.',
          'error'
        );
      }
    } finally {
      busy = false;
      input.disabled = false;
      updateSendButton();
      input.focus();
    }
  }
  async function loadDocuments() {
    if (!documentList) return;
    try {
      const response = await fetch(
        apiUrl('/api/documents/')
      );
      if (!response.ok) {
        throw new Error(
          `Documents request failed (${response.status})`
        );
      }
      const data = await response.json();
      const files = Array.isArray(data.documents)
        ? data.documents
        : [];
      if (documentCount) {
        documentCount.textContent = files.length;
      }
      if (files.length === 0) {
        documentList.innerHTML =
          '<p class="empty">No documents uploaded yet.</p>';
        return;
      }
      documentList.innerHTML = '';
      files.forEach((file) => {
        const item = document.createElement('div');
        item.className = 'document-item';
        const avatar = document.createElement('span');
        avatar.className = 'message-avatar';
        avatar.textContent = '⌁';
        const fileInfo = document.createElement('span');
        fileInfo.className = 'document-file';
        const strong = document.createElement('strong');
        strong.textContent =
          file.filename || 'Unnamed file';
        const small = document.createElement('small');
        const size =
          Number(file.size || 0) / 1024;
        const type =
          file.type
            ? String(file.type).toUpperCase()
            : 'FILE';
        small.textContent =
          `${size.toFixed(1)} KB · ${type}`;
        fileInfo.appendChild(strong);
        fileInfo.appendChild(small);
        const deleteButton =
          document.createElement('button');
        deleteButton.className = 'delete-file';
        deleteButton.type = 'button';
        deleteButton.textContent = '×';
        deleteButton.dataset.file =
          file.filename || '';
        deleteButton.setAttribute(
          'aria-label',
          `Delete ${file.filename || 'file'}`
        );
        item.appendChild(avatar);
        item.appendChild(fileInfo);
        item.appendChild(deleteButton);
        documentList.appendChild(item);
      });
    } catch (error) {
      console.error(
        'Load documents error:',
        error
      );
      documentList.innerHTML =
        '<p class="empty">API offline.</p>';
      if (documentCount) {
        documentCount.textContent = '0';
      }
    }
  }
  async function loadHistory() {
    if (!conversationList) return;
    try {
      const response = await fetch(
        apiUrl('/api/chat/history')
      );
      if (!response.ok) {
        throw new Error(
          `History request failed (${response.status})`
        );
      }
      const data = await response.json();
      const conversations =
        Array.isArray(data.conversations)
          ? data.conversations
          : [];
      if (conversations.length === 0) {
        conversationList.innerHTML =
          '<p class="empty">Your chats will appear here.</p>';
        return;
      }
      conversationList.innerHTML = '';
      conversations
        .slice(0, 12)
        .forEach((conversation) => {
          const button =
            document.createElement('button');
          button.type = 'button';
          button.className = 'conversation';
          if (
            conversation.conversation_id ===
            conversationId
          ) {
            button.classList.add('active');
          }
          button.dataset.id =
            conversation.conversation_id || '';
          button.textContent =
            conversation.title ||
            'Untitled conversation';
          conversationList.appendChild(button);
        });
    } catch (error) {
      console.error(
        'Load history error:',
        error
      );
      conversationList.innerHTML =
        '<p class="empty">API offline.</p>';
    }
  }
  async function loadConversationMessages(id) {
    if (!id || !chat) return;
    try {
      const response = await fetch(
        apiUrl(
          `/api/chat/history/${encodeURIComponent(id)}`
        )
      );
      if (!response.ok) return;
      const data = await response.json();
      const messagesList =
        Array.isArray(data.messages)
          ? data.messages
          : [];
      chat.innerHTML = '';
      if (messagesList.length === 0) {
        if (welcome) {
          chat.appendChild(welcome);
          welcome.hidden = false;
        }
        return;
      }
      if (welcome) {
        welcome.hidden = true;
      }
      messagesList.forEach((msg) => {
        message(msg.content, msg.role);
      });
    } catch (error) {
      console.error(
        'Failed to load conversation messages:',
        error
      );
    }
  }
  function createNewChat() {
    conversationId = createConversationId();
    if (chat) {
      chat.innerHTML = '';
      if (welcome) {
        chat.appendChild(welcome);
        welcome.hidden = false;
      }
    }
    if (chatTitle) {
      chatTitle.textContent = 'New conversation';
    }
    loadHistory();
    if (input) {
      input.value = '';
      input.focus();
      resizeInput();
    }
    updateSendButton();
  }
  async function uploadDocument(file) {
    if (!file) return;
    if (!uploadStatus) return;
    uploadStatus.className = 'upload-status active';
    uploadStatus.textContent =
      `Uploading ${file.name}…`;
    if (sendBtn) sendBtn.disabled = true;
    if (fileInput) fileInput.disabled = true;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(
        apiUrl('/api/documents/upload'),
        {
          method: 'POST',
          body: formData
        }
      );
      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }
      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          'Upload failed'
        );
      }
      uploadStatus.textContent =
        `Processing and indexing ${file.name}…`;
      await loadDocuments();
      uploadStatus.className = 'upload-status';
      uploadStatus.textContent =
        `${file.name} indexed successfully.`;
      const uploadedFilesDetails = document.getElementById('uploaded-files');
      if (uploadedFilesDetails) {
        uploadedFilesDetails.open = true;
      }
    } catch (error) {
      console.error(
        'Upload error:',
        error
      );
      uploadStatus.className =
        'upload-status error';
      uploadStatus.textContent =
        error.message || 'Document upload failed.';
    } finally {
      if (sendBtn) {
        sendBtn.disabled = false;
        updateSendButton();
      }
      if (fileInput) fileInput.disabled = false;
    }
  }
  async function deleteDocument(filename) {
    if (!filename) return;
    const confirmed = confirm(
      `Delete ${filename}?`
    );
    if (!confirmed) return;
    try {
      const response = await fetch(
        apiUrl(
          `/api/documents/${encodeURIComponent(filename)}`
        ),
        {
          method: 'DELETE'
        }
      );
      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }
      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          'Could not delete document.'
        );
      }
      await loadDocuments();
    } catch (error) {
      console.error(
        'Delete document error:',
        error
      );
      message(
        error.message ||
        'Could not delete document.',
        'error'
      );
    }
  }
  async function clearCurrentChat() {
    const confirmed = confirm(
      'Clear this chat? Uploaded documents will stay available.'
    );
    if (!confirmed) return;
    try {
      const response = await fetch(
        apiUrl(
          `/api/chat/${encodeURIComponent(conversationId)}`
        ),
        {
          method: 'DELETE'
        }
      );
      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }
      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.message ||
          'Could not clear chat.'
        );
      }
    } catch (error) {
      console.error(
        'Clear chat error:',
        error
      );
      message(
        error.message ||
        'Could not clear chat.',
        'error'
      );
      return;
    }
    createNewChat();
  }
  if (chatForm) {
    chatForm.addEventListener(
      'submit',
      (event) => {
        event.preventDefault();
        const text =
          input?.value.trim() || '';
        send(text);
      }
    );
  }
  if (input) {
    input.addEventListener(
      'input',
      () => {
        resizeInput();
        updateSendButton();
      }
    );
    input.addEventListener(
      'keydown',
      (event) => {
        if (
          event.key === 'Enter' &&
          !event.shiftKey
        ) {
          event.preventDefault();
          if (chatForm) {
            chatForm.requestSubmit();
          }
        }
      }
    );
  }
  if (suggestions) {
    suggestions.addEventListener(
      'click',
      (event) => {
        const button =
          event.target.closest('button');
        if (!button) return;
        const text =
          button.textContent.trim();
        if (text) {
          send(text);
        }
      }
    );
  }
  if (newChatBtn) {
    newChatBtn.addEventListener(
      'click',
      createNewChat
    );
  }
  if (fileInput) {
    fileInput.addEventListener(
      'change',
      async (event) => {
        const file =
          event.target.files?.[0];
        if (!file) return;
        await uploadDocument(file);
        event.target.value = '';
      }
    );
  }
  if (documentList) {
    documentList.addEventListener(
      'click',
      async (event) => {
        const button =
          event.target.closest(
            '[data-file]'
          );
        if (!button) return;
        const filename =
          button.dataset.file;
        await deleteDocument(filename);
      }
    );
  }
  if (clearChatBtn) {
    clearChatBtn.addEventListener(
      'click',
      clearCurrentChat
    );
  }
  if (conversationList) {
    conversationList.addEventListener(
      'click',
      async (event) => {
        const button =
          event.target.closest(
            '.conversation'
          );
        if (!button) return;
        const id =
          button.dataset.id;
        if (!id) return;
        conversationId = id;
        if (chatTitle) {
          chatTitle.textContent =
            button.textContent ||
            'Conversation';
        }
        await loadConversationMessages(id);
        await loadHistory();
        if (input) {
          input.focus();
        }
      }
    );
  }
  resizeInput();
  updateSendButton();
  loadDocuments();
  loadHistory();
  if (input) {
    input.focus();
  }
  console.log(
    'RAG Chatbot initialized.'
  );
  console.log(
    'API Base:',
    BASE
  );
  console.log(
    'Conversation ID:',
    conversationId
  );
});

if (typeof window !== 'undefined') {
  window.AppMarkdown = {
    splitMarkdownRow,
    isTableRow,
    isTableSeparator
  };
}
