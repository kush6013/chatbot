document.addEventListener('DOMContentLoaded', () => {
  const isLocal = ['localhost', '127.0.0.1'].includes(location.hostname);
  const BASE =
    window.RAG_API_BASE ||
    (isLocal
      ? 'http://127.0.0.1:8000'
      : 'https://chatbot-exuw.onrender.com');
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
    const textElement = document.createElement('div');
    textElement.textContent = text ?? '';
    body.appendChild(textElement);
    if (Array.isArray(sources) && sources.length > 0) {
      const sourcesContainer = document.createElement('div');
      sourcesContainer.className = 'sources';
      sources.forEach((source) => {
        const sourceElement = document.createElement('span');
        sourceElement.className = 'source';
        const sourceName = source?.source ?? 'Unknown source';
        const page = source?.page ?? '?';
        sourceElement.textContent = `${sourceName} · p. ${page}`;
        sourcesContainer.appendChild(sourceElement);
      });
      body.appendChild(sourcesContainer);
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
    article.innerHTML = `
      <div class="message-avatar">✦</div>
      <div class="thinking">
        <i></i>
        <i></i>
        <i></i>
      </div>
    `;
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
    uploadStatus.className = 'upload-status';
    uploadStatus.textContent =
      `Uploading ${file.name}…`;
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
        `${file.name} indexed successfully.`;
      const uploadedFilesDetails = document.getElementById('uploaded-files');
      if (uploadedFilesDetails) {
        uploadedFilesDetails.open = true;
      }
      await loadDocuments();
    } catch (error) {
      console.error(
        'Upload error:',
        error
      );
      uploadStatus.className =
        'upload-status error';
      uploadStatus.textContent =
        error.message ||
        'Document upload failed.';
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
